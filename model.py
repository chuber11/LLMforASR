
import torch
from torch import nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
from types import MethodType

def printms(t):
    print(t.mean().item(),t.std().item())

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
        #printms(hidden_states)
        #printms(audio_features)
        hidden_states = torch.cat([hidden_states[:,:1],
                                   audio_features.to(hidden_states.dtype),
                                   hidden_states[:,1:]],1)
        attention_mask = torch.cat([attention_mask[:,:1],
                                    torch.ones(audio_features.shape[:2],dtype=attention_mask.dtype, device=attention_mask.device),
                                    attention_mask[:,1:]],1)
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

class ASRModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", flash_attn=True, flash_rotary=True, fused_dense=True, device_map="cuda", trust_remote_code=True)

        self.decoder.forward = MethodType(forward_llm, self.decoder) # Ugly but works
        self.decoder.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, self.decoder)

        for p in self.decoder.parameters(): # Freeze decoder
            p.requires_grad = False

        self.audio_encoder = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype="auto", device_map="cuda")

        for p in self.audio_encoder.parameters(): # Freeze audio encoder
            p.requires_grad = False

        self.bridge_network = None

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

        self.prompt = "This was an audio recording. I think the transcript is: "

        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())/1000000:.0f} M, number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1000000:.0f} M")

    def encode_audio(self, wav_samples):
        audio_features = 0.03*torch.randn(len(wav_samples),100,2560,device="cuda") # TODO
        return audio_features

    def forward(self, wav_samples, text_labels=None):
        if text_labels is not None:
            pass
        else:
            inputs = self.tokenizer([self.prompt for _ in wav_samples], return_tensors="pt", return_attention_mask=False).to("cuda")
            inputs["audio_features"] = self.encode_audio(wav_samples)
            
            outputs = self.decoder.generate(**inputs, max_new_tokens=100)
            text = self.tokenizer.batch_decode(outputs)

            return text

if __name__ == "__main__":
    model = ASRModel()

    transcript = model([0,1])

    for t in transcript:
        print(t)

