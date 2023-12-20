
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import WhisperForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache

import os

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, List
from types import MethodType

from torch.cuda.amp import autocast

def printms(s,t):
    print(s,t.shape,t.mean().item(),t.std().item())

def forward_llm(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    audio_features = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if audio_features is not None:
        input_length = input_ids.shape[1]
        input_ids[input_ids==1] = 0
        input_ids = torch.cat([self.pre_prompt_tokens.expand(input_ids.shape[0],-1),
                               self.post_prompt_tokens.expand(input_ids.shape[0],-1),
                               input_ids[:,1:-1]],1)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        audio_features = 0.045*audio_features

        #printms("I",inputs_embeds)
        #printms("A",audio_features)

        inputs_embeds = torch.cat([inputs_embeds[:,:self.pre_prompt_tokens.shape[1]],
                                   audio_features,
                                   inputs_embeds[:,self.pre_prompt_tokens.shape[1]:]],1)

        attention_mask = torch.cat([torch.ones(attention_mask.shape[0],self.pre_prompt_tokens.shape[1]+audio_features.shape[1]+self.post_prompt_tokens.shape[1], dtype=attention_mask.dtype, device=attention_mask.device),
                                    attention_mask[:,1:-1]],1)

        input_ids = None
    else:
        input_ids = input_ids[:,-1:]

    if position_ids is None:
        if past_key_values is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            past_length = past_key_values[0][0].shape[2]
            position_ids = torch.full_like(input_ids, past_length)
            attention_mask = torch.ones(input_ids.shape[0],past_length+1,dtype=torch.bool,device=input_ids.device)

    #print(input_ids.shape if input_ids is not None else inputs_embeds.shape, attention_mask.shape, position_ids.shape)

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., -(input_length-2):, :].contiguous()
        shift_labels = labels[..., 2:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=0)
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def prepare_inputs_for_generation(
    self, input_ids, audio_features=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    # Omit tokens covered by past_key_values
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

        audio_features = None

    """position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]"""

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            #"position_ids": position_ids,
            "audio_features": audio_features,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

class BridgeNetwork(nn.Module):
    def __init__(self, num_layers, dim_h):
        super().__init__()

        dim_in = 1280
        dim_out = 4096

        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(dim_h if i!=0 else dim_in,
                                    dim_h if i!=num_layers-1 else dim_out,
                                    4,
                                    2))

        self.layers = nn.ModuleList(layers)

    def forward(self, inp):
        with autocast(enabled=True):
            out = self.layers[0](inp.transpose(2,1))
            for layer in self.layers[1:]:
                out = layer(F.gelu(out))
            out = out.transpose(2,1)
        return out

class ASRModelConfig(PretrainedConfig):
    def __init__(self, *args, **kwargs):
        self.decoder_name="mistralai/Mistral-7B-Instruct-v0.2"
        self.audio_encoder_name="openai/whisper-large-v3"
        self.bridge_layers = 3
        self.bridge_dim = 4096

        super().__init__(*args, **kwargs)

class ASRModel(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = ASRModelConfig

    def __init__(self, config=None, tokenizer=None):
        if config is None:
            config = ASRModelConfig()
        super().__init__(config)

        self.decoder = AutoModelForCausalLM.from_pretrained(config.decoder_name, torch_dtype="auto", device_map="cuda")

        self.decoder.forward = MethodType(forward_llm, self.decoder) # Ugly but works
        self.decoder.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, self.decoder)

        for p in self.decoder.parameters(): # Freeze decoder
            p.requires_grad = False

        self.audio_encoder = WhisperForConditionalGeneration.from_pretrained(config.audio_encoder_name, torch_dtype="auto", device_map="cuda").model.encoder

        for p in self.audio_encoder.parameters(): # Freeze audio encoder
            p.requires_grad = False

        self.bridge_network = BridgeNetwork(config.bridge_layers,config.bridge_dim).to("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(config.decoder_name) if tokenizer is None else tokenizer
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.set_pre_prompt()
        self.set_post_prompt()

        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())/1000000:.0f} M, number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1000000:.0f} M")

    def set_pre_prompt(self, pre_prompt="[INST]"):
        self.decoder.pre_prompt_tokens = self.tokenizer([pre_prompt], return_tensors="pt", return_attention_mask=False).to("cuda")["input_ids"]

    def set_post_prompt(self, post_prompt="Transcribe the given audio recording. [/INST]"):
        self.decoder.post_prompt_tokens = self.tokenizer([post_prompt], return_tensors="pt", return_attention_mask=False).to("cuda")["input_ids"][:,1:]

    def encode_audio(self, audio_features):
        audio_encoding = self.audio_encoder(audio_features).last_hidden_state
        audio_encoding = self.bridge_network(audio_encoding)
        return audio_encoding

    def forward(self, audio_features, text_labels):
        audio_features = self.encode_audio(audio_features)

        text_labels["audio_features"] = audio_features
        text_labels["labels"] = text_labels["input_ids"]

        prediction = self.decoder(**text_labels)

        return prediction

    def inference(self, inputs):
        if "input_ids" not in inputs:
            inputs["input_ids"] = torch.ones(inputs["audio_features"].shape[0], 1, device="cuda", dtype=torch.int64)
        else:
            print("IN",self.tokenizer.batch_decode(inputs["input_ids"])[0])

        inputs["audio_features"] = self.encode_audio(inputs["audio_features"]) #.half())

        print(inputs.keys())
        
        outputs = self.decoder.generate(**inputs, max_new_tokens=100, no_repeat_ngram_size=6)
        text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        #print("OUT",self.tokenizer.batch_decode(outputs)[0])

        #text = [t if not t.startswith("<unk> ") else t[len("<unk> "):] for t in text]

        return text

    def save_pretrained(self, *args, **kwargs):
        state_dict = {k:v for k,v in self.state_dict().items() if k.startswith("bridge_network")}
        super().save_pretrained(*args, state_dict=state_dict, **kwargs)

