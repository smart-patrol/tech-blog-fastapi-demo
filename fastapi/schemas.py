from typing import List
from pydantic import BaseModel
from enum import Enum


class RequestBody(BaseModel):
    file: str

class ResponseValues(str, Enum):
    hockey = "rec.sport.hockey"
    space = "sci.space"
    politics = "talk.politics.misc"


class ResponseBody(BaseModel):
    class_id: str
    class_name: str


class LabelResponseBody(BaseModel):
    label: str
    probabilities: List[float]
