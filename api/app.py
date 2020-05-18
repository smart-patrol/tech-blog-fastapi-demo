import logging
import os
import pickle

import numpy as np

from aiohttp import ClientSession
from core import security
from fastapi import Depends, FastAPI
from schemas import (
    HearbeatResult,
    LabelResponseBody,
    RequestBody,
    ResponseBody,
    ResponseValues,
    TextSample,
)

logger = logging.getLogger("LOGGER")
logger.setLevel(logging.ERROR)


app = FastAPI(
    title="simple-model",
    description="a simple model-serving skateboard in FastAPI",
    version="0.1",
)

with open(os.getenv("MODEL_PATH"), "rb") as rf:
    clf = pickle.load(rf)

client_session = ClientSession()


@app.get("/healthcheck", response_model=HearbeatResult, name="healtcheck")
def get_hearbeat() -> HearbeatResult:
    heartbeat = HearbeatResult(is_alive=True)
    return heartbeat


# On Aysnc: https://fastapi.tiangolo.com/async/
@app.post("/predict", response_model=ResponseBody, tags=["predict"])
async def predict(
    body: RequestBody, authenticated: bool = Depends(security.validate_request),
):
    data = np.array(body.to_array())

    probas = await clf.predict_proba(data)
    predictions = await probas.argmax(axis=1)

    logger.info(predictions)

    return {
        "predictions": (
            np.tile(clf.classes_, (len(predictions), 1))[
                np.arange(len(predictions)), predictions
            ].tolist()
        ),
        "probabilities": probas[np.arange(len(predictions)), predictions].tolist(),
    }


@app.post("/predict/{label}", response_model=LabelResponseBody, tags=["predict_label"])
async def predict_label(
    label: ResponseValues,
    body: RequestBody,
    authenticated: bool = Depends(security.validate_request),
):

    data = np.array(body.to_array())

    probas = await clf.predict_proba(data)
    target_idx = await clf.classes_.tolist().index(label.value)

    logger.info(probas)

    return {"label": label.value, "probabilities": probas[:, target_idx].tolist()}


@app.on_event("shutdown")
async def cleanup():
    await client_session.close()
