from enum import Enum
from typing import List

from pydantic import BaseModel


class TextSample(BaseModel):
    text: str


class RequestBody(BaseModel):
    samples: List[TextSample]

    def to_array(self):
        return [sample.text for sample in self.samples]


class ResponseValues(str, Enum):
    # https://www.python.org/dev/peps/pep-0435/
    hockey = "rec.sport.hockey"
    space = "sci.space"
    politics = "talk.politics.misc"


class ResponseBody(BaseModel):
    predictions: List[str]
    probabilities: List[float]


class LabelResponseBody(BaseModel):
    label: str
    probabilities: List[float]


class HearbeatResult(BaseModel):
    is_alive: bool