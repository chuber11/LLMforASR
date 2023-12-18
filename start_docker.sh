
docker run --gpus all -it --rm --network=ltpipeline_LTPipeline -v /export/data2/chuber/2024/WhisperE+Phi2:/workspace/WhisperE+Phi2 -v /export/data2:/export/data2:ro -v /project/asr_systems:/project/asr_systems:ro -v /project/OML:/project/OML:ro nvcr.io/nvidia/pytorch:22.04-py3

