
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput
from data import MyDataset
from tqdm import tqdm
import torch
import os

model_path = "openai/whisper-large-v3"

device = "cuda"

processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path, device_map=device, torch_dtype="auto")

segfile = "data_test/*.test.seg.aligned"
dataset = MyDataset(segfiles=segfile)

batch_size = 16

outputfile = "hypos/hypo_whisper_norepeat6.txt"
if os.path.isfile(outputfile):
    print("Output file already exists, continue?")
    breakpoint()

with open(outputfile, "w") as f:
    for i in tqdm(range(0,len(dataset),batch_size)):
        data = [dataset[j] for j in range(i,min(len(dataset),i+batch_size))]

        input_values = processor([d["audio"] for d in data],sampling_rate=16000, return_tensors="pt")["input_features"].to(device).to(torch.float16)

        ids = [d["id"] for d in data]

        output = model.generate(
            input_values, 
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        ) # Predicts the language

        input_languages = ["en","de"]

        predictable_ids = [id for id,token in processor.tokenizer.added_tokens_decoder.items() if token.content[2:-2] in input_languages]

        predicted_ids_small = output.scores[0][:,predictable_ids].argmax(-1)
        predicted_ids = torch.as_tensor(predictable_ids, device=predicted_ids_small.device)[predicted_ids_small]

        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

        pred_to_indices = {}
        for i,pred in enumerate(predicted_ids.tolist()):
            if pred not in pred_to_indices:
                pred_to_indices[pred] = [i]
            else:
                pred_to_indices[pred].append(i)

        outputs = {}
        for pred, indices in pred_to_indices.items():
            inputs = input_values[indices]
            forced_decoder_ids[0] = (forced_decoder_ids[0][0],pred)
            encoder_outputs = BaseModelOutput(last_hidden_state=output["encoder_hidden_states"][-1][indices])

            predicted_ids2 = model.generate(
                inputs, 
                no_repeat_ngram_size=6,
                forced_decoder_ids=forced_decoder_ids,
                encoder_outputs=encoder_outputs,
            )

            for o,i in zip(processor.batch_decode(predicted_ids2, skip_special_tokens=True),indices):
                outputs[i] = o
        
        outputs = [outputs[i] for i in range(len(outputs))]

        for t,id in zip(outputs, ids):
            t = t.replace("\n"," ").strip()
            print(id,t)
            f.write(id+" "+t+"\n")
