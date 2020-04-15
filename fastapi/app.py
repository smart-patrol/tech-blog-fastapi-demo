import os
import pickle
import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

from fastapi import FastAPI

from aiohttp import ClientSession

from schemas import ResponseBody, RequestBody
#    LabelResponseBody,
#    ResponseValues,
#    TextSample,
#)

app = FastAPI(
    title="image-net-classifier",
    description="a model to classify images",
    version="0.1",
)

imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.resnet18(pretrained=True)
model.eval()

from helpers import get_prediction,transform_image

client_session = ClientSession()


@app.get("/healthcheck")
async def healthcheck():
    msg = (
        "this sentence is already halfway over, "
        "and still hasn't said anything at all"
    )
    return {"message": msg}


@app.post("/predict", response_model=ResponseBody)
async def predict(body: RequestBody):

    img_bytes = body.file.read()
    class_id, class_name = get_prediction(image_bytes=img_bytes)
    
    return {'class_id': class_id, 'class_name': class_name}

@app.on_event("shutdown")
async def cleanup():
    await client_session.close()


