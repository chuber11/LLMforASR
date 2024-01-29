
import torch
#import torchaudio
import soundfile as sf
#import librosa
from torch.utils.data import Dataset

from glob import glob

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import math
import random

class MyDataset(Dataset):
    def __init__(self, dev=False, segfiles=None, replace=None, max_len=2000, augment=False):
        if segfiles is None:
            segfiles = "data/*.train.seg.aligned"
            #segfiles = "/project/OML/chuber/2022/NMTGMinor/exp/ASR-NW/data/orig_en_cased/cv.train.seg.aligned"
            #segfiles = "/project/OML/chuber/2022/NMTGMinor/exp/ASR-NW/data/orig_en_cased/*.train.seg.aligned"

        if dev:
            segfiles = segfiles.replace("train","dev")

        if augment:
            segfiles = segfiles.replace("data","data_augment")

        if replace is None:
            replace = [("/project/asr_systems/LT2021/EN/data","/export/data2/chuber/ASR/data/EN")]

        self.ids = []
        self.audio_paths = []
        self.timestamps = []
        self.labels = []
        for segfile in glob(segfiles):
            print(segfile)
            labelfile = ".".join(segfile.split(".")[:-2])+".cased"
            for line, line2 in zip(open(segfile),open(labelfile)):
                line = line.strip().split()
                self.ids.append(line[0])
                audio_path = line[1]
                for r in replace:
                    audio_path = audio_path.replace(r[0],r[1])
                self.audio_paths.append(audio_path)
                if len(line) == 2:
                    self.timestamps.append(None)
                elif len(line) == 4:
                    self.timestamps.append((float(line[2]),float(line[3])))
                else:
                    raise RuntimeError
                self.labels.append(line2.strip())

                #if len(self.audio_paths) >= 16*3:
                #    break

        random.seed(42)

        combined_lists = list(zip(self.ids, self.audio_paths, self.timestamps, self.labels))
        random.shuffle(combined_lists)
        self.ids, self.audio_paths, self.timestamps, self.labels = zip(*combined_lists)

        self.len = len(self.audio_paths)
        if dev:
            self.len = min(max_len,self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.timestamps[idx] is not None:
            start, end = self.timestamps[idx]
            audio, sr = sf.read(self.audio_paths[idx], start=int(16000*start),stop=int(16000*end))
            #audio, sr  = torchaudio.load(self.audio_paths[idx], frame_offset=int(16000*start),num_frames=int(16000*(end-start)))
            #audio = audio[0]
        else:
            #audio, sr = torchaudio.load(self.audio_paths[idx])
            audio, sr = sf.read(self.audio_paths[idx])
            #audio, sr = librosa.load(self.audio_paths[idx])
        #sample = {"audio":audio[0].numpy(),"labels":self.labels[idx]}
        sample = {"audio":audio,"labels":self.labels[idx], "id":self.ids[idx]}
        return sample

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    tokenizer: Any
    return_ids: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]], inference=False) -> Dict[str, torch.Tensor]:

        audio = torch.cat([self.processor(item["audio"], sampling_rate=16000, return_tensors="pt").input_features for item in features], dim=0)
        text_labels = self.tokenizer([feature["labels"]+self.tokenizer.eos_token for feature in features], return_tensors="pt", padding=True)

        input_ids = text_labels["input_ids"]
        if not inference:
            text_labels["attention_mask"][input_ids==1] = 0
            input_ids[input_ids==1] = 0 # 1 = sos_token is added via pre prompt
        text_labels["input_ids"] = input_ids[:,:-1]
        text_labels["attention_mask"] = text_labels["attention_mask"][:,:-1]
        text_labels["labels"] = input_ids[:,1:]

        batch = {"audio_features": audio, "text_labels":text_labels}
        if self.return_ids:
            batch["ids"] = [item["id"] for item in features]

        return batch

def compute_metrics(pred):
    print(pred)
    breakpoint()
    return {}

    breakpoint()
    loss = pred.loss
    ppl = math.exp(loss.sum()/loss.shape[0])
    return {"ppl": ppl}

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"ppl":ppl, "wer": wer}

