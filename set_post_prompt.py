
import requests
import sys

requests.post("http://192.168.0.60:5000/asr/set_post_prompt", {"post_prompt": sys.argv[1]+" [/INST]"})
