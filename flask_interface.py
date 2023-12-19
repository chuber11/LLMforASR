# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from flask import Flask, request
import torch
import numpy as np
import math
import sys
import json
import threading
import queue
import uuid
import traceback

from model import ASRModel
from transformers import WhisperProcessor
from torch.cuda.amp import autocast

host = "0.0.0.0"
port = 5000

app = Flask(__name__)

def create_unique_list(my_list):
    my_list = list(set(my_list))
    return my_list

def initialize_model():
    model = ASRModel()
    model.load()
    
    audio_encoder_name = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(audio_encoder_name)

    print("ASR initialized")

    max_batch_size = 8

    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, processor, max_batch_size

def infer_batch(audio_wavs, prefix="", input_language="en", audio_sample_rate=16000):
    # get device based on the model parameters
    device = next(model.parameters()).device

    #data = {"audio_features": audio_features, "attention_mask": attention_mask}
    data = processor([w.numpy() for w in audio_wavs], sampling_rate=16000, return_tensors="pt")
    data = {"audio_features": data["input_features"].cuda()}

    if prefix != "":
        data["input_ids"] = model.tokenizer(prefix,return_tensors="pt")["input_ids"].to(device).expand(len(audio_wavs),-1)

    with autocast(enabled=True):
        text_output_raw = model.inference(data)

    print(text_output_raw[0])
    return text_output_raw

def use_model(reqs):

    if len(reqs) == 1:
        req = reqs[0]
        audio_tensor, prefix, input_language, output_language = req.get_data()
        if not (input_language == output_language or output_language == 'en'):
            result = {"hypo": "", "status":400, "message": 'Wrong option. Perform X->X "transcribe" or X->English "translate". Found {} -> {}'.format(input_language, output_language)}
            req.publish(result)
            return
        
        hypo = infer_batch(audio_wavs=[audio_tensor], input_language=input_language, prefix=prefix)[0]
            
        result = {"hypo": hypo.strip()}
        req.publish(result)

    else:
        audio_tensors = list()
        prefixes = list()
        input_languages = list()
        output_languages = list()

        batch_runnable = False

        for req in reqs:
            audio_tensor, prefix, input_language, output_language = req.get_data()
            audio_tensors.append(audio_tensor)
            prefixes.append(prefix)
            input_languages.append(input_language)
            output_languages.append(output_language)

        unique_prefix_list = create_unique_list(prefixes)
        unique_input_languages = create_unique_list(input_languages)
        unique_output_languages = create_unique_list(output_languages)
        if len(unique_prefix_list) == 1 and len(unique_input_languages) == 1 and len(unique_output_languages) == 1:
            batch_runnable = True

        if batch_runnable:
            hypos = infer_batch(audio_wavs=audio_tensors, input_language=input_languages[0], prefix=prefixes[0])

            for req, hypo in zip(reqs, hypos):
                result = {"hypo": hypo.strip()}
                req.publish(result)
        else:
            for req, audio_tensor, prefix, input_language, output_language \
                    in zip(reqs, audio_tensors, prefixes, input_languages, output_languages):
                hypo = infer_batch(audio_wavs=[audio_tensor], input_language=input_language, prefix=prefix)[0]                    
                result = {"hypo": hypo.strip()}
                req.publish(result)

def run_decoding():
    while True:
        reqs = [queue_in.get()]
        while not queue_in.empty() and len(reqs) < max_batch_size:
            req = queue_in.get()
            reqs.append(req)
            if req.priority >= 1:
                break

        print("Batch size:",len(reqs),"Queue size:",queue_in.qsize())

        try:
            use_model(reqs)
        except Exception as e:
            print("An error occured during model inference")
            traceback.print_exc()
            for req in reqs:
                req.publish({"hypo":"", "status":400})

class Priority:
    next_index = 0

    def __init__(self, priority, id, condition, data):
        self.index = Priority.next_index

        Priority.next_index += 1

        self.priority = priority
        self.id = id
        self.condition = condition
        self.data = data

    def __lt__(self, other):
        return (-self.priority, self.index) < (-other.priority, other.index)

    def get_data(self):
        return self.data

    def publish(self, result):
        dict_out[self.id] = result
        try:
            with self.condition:
                self.condition.notify()
        except:
            print("ERROR: Count not publish result")

def pcm_s16le_to_tensor(pcm_s16le):
    audio_tensor = np.frombuffer(pcm_s16le, dtype=np.int16)
    audio_tensor = torch.from_numpy(audio_tensor)
    audio_tensor = audio_tensor.float() / math.pow(2, 15)
    audio_tensor = audio_tensor.unsqueeze(1)  # shape: frames x 1 (1 channel)
    return audio_tensor

# corresponds to an asr_server "http://$host:$port/asr/infer/en,en" in StreamASR.py
# use None when no input- or output language should be specified
@app.route("/asr/infer/<input_language>,<output_language>", methods=["POST"])
def inference(input_language, output_language):
    pcm_s16le: bytes = request.files.get("pcm_s16le").read()
    prefix = request.files.get("prefix") # can be None
    if prefix is not None:
        prefix: str = prefix.read().decode("utf-8")

    # calculate features corresponding to a torchaudio.load(filepath) call
    audio_tensor = pcm_s16le_to_tensor(pcm_s16le).squeeze()

    priority = request.files.get("priority") # can be None
    try:
        priority = int(priority.read()) # used together with priority queue
    except:
        priority = 0

    condition = threading.Condition()
    with condition:
        id = str(uuid.uuid4())
        data = (audio_tensor,prefix,input_language,output_language)

        queue_in.put(Priority(priority,id,condition,data))

        condition.wait()

    result = dict_out.pop(id)
    status = 200
    if status in result:
        status = result.pop(status)

    # result has to contain a key "hypo" with a string as value (other optional keys are possible)
    return json.dumps(result), status

# called during automatic evaluation of the pipeline to store worker information
@app.route("/asr/version", methods=["POST"])
def version():
    # return dict or string (as first argument)
    return "WhisperEncoder+Mistral7BDecoder", 200

@app.route("/asr/set_pre_prompt", methods=["GET","POST"])
def set_pre_prompt():
    pre_prompt = request.form["pre_prompt"]
    print("Setting pre prompt:",pre_prompt)
    model.set_post_prompt(pre_prompt)
    return "DONE", 200

@app.route("/asr/set_post_prompt", methods=["GET","POST"])
def set_post_prompt():
    post_prompt = request.form["post_prompt"]
    print("Setting post prompt:",post_prompt)
    model.set_post_prompt(post_prompt)
    return "DONE", 200

model, processor, max_batch_size = initialize_model()

queue_in = queue.PriorityQueue()
dict_out = {}

decoding = threading.Thread(target=run_decoding)
decoding.daemon = True
decoding.start()


if __name__ == "__main__":
    app.run(host=host, port=port)
