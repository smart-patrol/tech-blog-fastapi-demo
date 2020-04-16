import os
import pickle
import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

from fastapi import FastAPI, File, UploadFile

from aiohttp import ClientSession

#from schemas import ResponseBody, RequestBody
#    LabelResponseBody,
#    ResponseValues,
#    TextSample,
#)

app = FastAPI(
    title="image-net-classifier",
    description="a model to classify images",
    version="0.1",
)

#from helpers import get_prediction,transform_image

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(model, image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.resnet18(pretrained=True)
model.eval()

client_session = ClientSession()


@app.post("/predict/", tags=["predict"])
def predict(file: bytes = File(...)):
    class_id, class_name = get_prediction(model,image_bytes=file)
    return {'class_id': class_id, 'class_name': class_name}

# http:// X /predict/?file=""
@app.post("/predict_async/", tags=["predict_async"])
async def predict_async(file: bytes = File(...)):
    class_id, class_name = get_prediction(model,image_bytes=file)
    return {'class_id': class_id, 'class_name': class_name}

@app.on_event("shutdown")
async def cleanup():
    await client_session.close()