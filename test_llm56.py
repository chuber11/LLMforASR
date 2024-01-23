
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR, format='%(message)s')

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

while True:
    text = input()
    text = "[INST] "+text+" [/INST]"

    inputs = tokenizer(text, return_tensors="pt").to(0)

    outputs = model.generate(**inputs, max_new_tokens=1000) #, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #print("----------------- PROMPT -----------------")
    #print(text)
    print("----------------- OUTPUT -----------------")
    print(output[len(text):].strip())
    print("----------------- PROMPT -----------------")
