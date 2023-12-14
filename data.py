
import torch
import soundfile as sf
from torch.utils.data import Dataset

from glob import glob

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import math

class MyDataset(Dataset):
    def __init__(self, dev=False, segfiles=None, replace=None):
        if segfiles is None:
            segfiles = "cv.train.seg.aligned"
            #segfiles = "/project/OML/chuber/2022/NMTGMinor/exp/ASR-NW/data/orig_en_cased/cv.train.seg.aligned"
            #segfiles = "/project/OML/chuber/2022/NMTGMinor/exp/ASR-NW/data/orig_en_cased/*.train.seg.aligned"

        if dev:
            segfiles = segfiles.replace("train","dev")

        if replace is None:
            replace = [("/project/asr_systems/LT2021/EN/data","/export/data2/chuber/ASR/data/EN")]

        self.audio_paths = []
        self.timestamps = []
        self.labels = []
        for segfile in glob(segfiles):
            print(segfile)
            labelfile = ".".join(segfile.split(".")[:-2])+".cased"
            for line, line2 in zip(open(segfile),open(labelfile)):
                line = line.strip().split()
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

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        #audio, sr = torchaudio.load(self.audio_paths[idx])
        audio, sr = sf.read(self.audio_paths[idx])
        if self.timestamps[idx] is not None: # TODO: only load relevant audio
            start, end = self.timestamps[idx]
            audio = audio[:,int(1600*start),int(16000*end)]
        #sample = {"audio":audio[0].numpy(),"labels":self.labels[idx]}
        sample = {"audio":audio,"labels":self.labels[idx]}
        return sample

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        audio = torch.cat([self.processor(item["audio"], sampling_rate=16000, return_tensors="pt").input_features for item in features], dim=0)
        input_ids = self.tokenizer([feature["labels"] for feature in features], return_tensors="pt", padding=True)

        batch = {"audio_features": audio, "text_labels":input_ids}

        return batch

def compute_metrics(pred):
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

