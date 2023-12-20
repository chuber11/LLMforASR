
import torch

from model import ASRModel
from data import MyDataset, DataCollatorSpeechSeq2SeqWithPadding

from transformers import WhisperProcessor

from torch.cuda.amp import autocast

from tqdm import tqdm

import sys
from glob import glob

path = sys.argv[1] if len(sys.argv) > 1 else ""

print("Using path",path)

model = ASRModel.from_pretrained(path)

dataset = MyDataset(segfiles="data_test/*.test.seg.aligned", dev=True)

audio_encoder_name = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(audio_encoder_name)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=model.tokenizer, return_ids=True)

batch_size = 8

with open("hypos/hypo.txt", "w") as f:
    for i in tqdm(range(0,len(dataset),batch_size)):
        data = data_collator([dataset[j] for j in range(i,min(len(dataset),i+batch_size))])
        ids = data.pop("ids")
        data = {k:v.to("cuda") for k,v in data.items() if k!="text_labels"}

        with autocast(enabled=True):
            transcript = model.inference(data)

        for t,id in zip(transcript, ids):
            print(id,t)
            t = t.replace("\n"," ")
            f.write(id+" "+t+"\n")

