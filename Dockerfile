FROM nvcr.io/nvidia/pytorch:23.01-py3

WORKDIR /code
RUN pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git
RUN pip install transformers timm accelerate
RUN pip install git+https://github.com/openai/CLIP.git

#Setup nanoowl module
RUN pip install git+https://github.com/NVIDIA-AI-IOT/nanoowl

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./src /code

CMD [ "python", "main.py" ]