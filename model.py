
import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
from types import MethodType

from torch.cuda.amp import autocast

def printms(s,t):
    print(s,t.shape,t.mean().item(),t.std().item())

def forward_llm(
    self,
    input_ids,
    audio_features,
    past_key_values = None,
    attention_mask = None,
    labels = None,
    **kwargs,
) -> CausalLMOutputWithPast:
    if audio_features is not None: # first forward call of generate
        hidden_states = self.transformer.embd(input_ids)
        #printms("H",hidden_states)
        #printms("A",audio_features)
        hidden_states = torch.cat([audio_features.to(hidden_states.dtype),
                                   hidden_states],1)
        attention_mask = torch.cat([torch.ones(audio_features.shape[:2],dtype=attention_mask.dtype, device=attention_mask.device),
                                    attention_mask],1)
    else:
        hidden_states = self.transformer.embd(input_ids)

    for layer in self.transformer.h:
        hidden_states = layer(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )

    lm_logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        lm_logits = lm_logits[:,audio_features.shape[1]:]
        loss = self.loss(lm_logits, labels)

    return CausalLMOutputWithPast(loss=loss, logits=lm_logits, past_key_values=past_key_values)

@dataclass
class InferenceParams:
    """Inference parameters passed to model to efficiently calculate
    and store context during inference.

    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/generation.py.

    Args:
        max_seqlen: Maximum sequence length.
        max_batch_size: Maximum batch size.
        seqlen_offset: Sequence length offset.
        batch_size_offset: Batch size offset.
        key_value_memory_dict: Key value memory dictionary.
        lengths_per_sample: Lengths per sample.

    """

    max_seqlen: int = field(metadata={"help": "Maximum sequence length."})

    max_batch_size: int = field(metadata={"help": "Maximum batch size."})

    seqlen_offset: int = field(default=0, metadata={"help": "Sequence length offset."})

    batch_size_offset: int = field(default=0, metadata={"help": "Batch size offset."})

    key_value_memory_dict: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Key value memory dictionary."}
    )

    lengths_per_sample: torch.Tensor = field(default=None, metadata={"help": "Lengths per sample."})

def prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    audio_features = None,
    past_key_values = None,
    attention_mask = None,
    **kwargs,
):
    if past_key_values is None or not (isinstance(past_key_values, InferenceParams)):
        past_key_values = InferenceParams(
            max_seqlen=self.config.n_positions,
            max_batch_size=input_ids.shape[0],
            seqlen_offset=0,
            batch_size_offset=0,
            key_value_memory_dict={},
            lengths_per_sample=None,
        )
    else:
        # Assume that `past_key_values` has cached all tokens up to the last token in `input_ids`
        past_key_values.seqlen_offset = input_ids.shape[1] - 1
        input_ids = input_ids[:, -1].unsqueeze(-1)
        audio_features = None

    return {
        "input_ids": input_ids,
        "audio_features": audio_features,
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
    }

class BridgeNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        dim_h = 1280
        
        self.cnn1 = nn.Conv1d(1280,dim_h,4,2)
        self.cnn2 = nn.Conv1d(dim_h,dim_h,4,2)
        self.cnn3 = nn.Conv1d(dim_h,2560,4,2)

    def forward(self, inp):
        with autocast(enabled=True):
            out = inp.transpose(2,1)
            out = self.cnn1(out)
            out = self.cnn2(F.gelu(out))
            out = self.cnn3(F.gelu(out))
            out = out.transpose(2,1)
        return out

class ASRModel(nn.Module):
    def __init__(self, decoder_name="microsoft/phi-2", audio_encoder_name="openai/whisper-large-v3", tokenizer=None):
        super().__init__()

        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name, torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map="cuda", trust_remote_code=True)

        self.decoder.forward = MethodType(forward_llm, self.decoder) # Ugly but works
        self.decoder.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, self.decoder)

        for p in self.decoder.parameters(): # Freeze decoder
            p.requires_grad = False

        self.audio_encoder = WhisperForConditionalGeneration.from_pretrained(audio_encoder_name, torch_dtype="auto", device_map="cuda").model.encoder

        for p in self.audio_encoder.parameters(): # Freeze audio encoder
            p.requires_grad = False

        self.bridge_network = BridgeNetwork().to("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name, trust_remote_code=True) if tokenizer is None else tokenizer
        self.prompt = "This was an audio recording. I think the transcript is as follows: "

        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())/1000000:.0f} M, number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1000000:.0f} M")

    def encode_audio(self, audio_features):
        audio_features = 0.8*self.audio_encoder(audio_features).last_hidden_state
        return self.bridge_network(audio_features)

    def forward(self, audio_features, text_labels=None):
        audio_features = self.encode_audio(audio_features)

        if text_labels is not None:
            text_labels["audio_features"] = audio_features
            text_labels["labels"] = text_labels["input_ids"]
            prediction = self.decoder(**text_labels)
            return prediction
        else:
            inputs = self.tokenizer([self.prompt for _ in audio_features], return_tensors="pt", return_attention_mask=False).to("cuda")
            inputs["audio_features"] = audio_features
            
            outputs = self.decoder.generate(**inputs, max_new_tokens=200)
            text = self.tokenizer.batch_decode(outputs)

            return text

if __name__ == "__main__":
    from data import MyDataset, DataCollatorSpeechSeq2SeqWithPadding

    model = ASRModel()
    model.tokenizer.pad_token = model.tokenizer.eos_token

    dataset = MyDataset(dev=True)

    audio_encoder_name = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(audio_encoder_name)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=model.tokenizer)

    data = data_collator([dataset[0],dataset[1]])

    transcript = model(data["audio_features"].to("cuda").half())

    for t in transcript:
        print(t)

