
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

text = "Hello my name is"

while True:
    text = "[INST] "+text+" [/INST]"
    inputs = tokenizer(text, return_tensors="pt").to(0)

    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    text = input("Enter next prompt: ")
