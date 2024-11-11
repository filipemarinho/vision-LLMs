from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
import uuid
from PIL import Image
from image_inference import ImageInferencePipeline

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Image inference API is running."}
