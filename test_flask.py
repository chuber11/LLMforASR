
import requests

use1 = True

if use1:
    wav = "/project/OML/chuber/2024/data/NumbersTestset/NumbersTestset__94/AudioData/numberstestset-usr0003.wav"
    segfile = "/project/OML/chuber/2024/data/NumbersTestset/NumbersTestset__94/numberstestset.stm"
else:
    wav = "/project/OML/chuber/2024/data/test.wav"
    segfile = "/project/OML/chuber/2024/data/test.stm"

wav = open(wav,"rb").read()[78:]

for line in open(segfile):
    if line[:2] == ";;":
        continue
    line = line.strip().split()
    start = float(line[4])
    end = float(line[5])
    label = " ".join(line[7:])

    wav_ = wav[int(32000*start):int(32000*end)]

    res = requests.post("http://192.168.0.60:5000/asr/infer/None,None", files={"pcm_s16le":wav_, "prefix": ""})
    hypo = res.json()["hypo"]

    print(hypo)

