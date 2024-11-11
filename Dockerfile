FROM nvcr.io/nvidia/pytorch:23.01-py3


RUN pip install pillow --upgrade 
RUN pip install tensorrt onnx
RUN pip install git+https://github.com/NVIDIA-AI-IOT/torch2trt.git
RUN pip install transformers timm accelerate
RUN pip install git+https://github.com/openai/CLIP.git

#Setup nanoowl module
RUN pip install git+https://github.com/NVIDIA-AI-IOT/nanoowl

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./src /code
WORKDIR /code
CMD [ "python", "image_inference.py" ]