from typing import Optional

import yaml
from pydantic import BaseModel, ValidationError

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_THRESHOLD = 50000
DEFAULT_TOLERANCE = 0.01


class PerceptronSettings(BaseModel):
    learning_rate: float = DEFAULT_LEARNING_RATE
    threshold: int = DEFAULT_THRESHOLD
    g: Optional[str]
    b: Optional[float]


class PerceptronConfig(BaseModel):
    type: str
    settings: PerceptronSettings


class TrainingValues(BaseModel):
    input: Optional[str]
    output: Optional[str]


class Config(BaseModel):
    training_values: Optional[TrainingValues] = dict()
    plot: Optional[bool] = False
    perceptron: PerceptronConfig


def get_config(config_file: str) -> Config:
    with open(config_file) as cf:
        config = yaml.safe_load(cf)["config"]
        try:
            return Config(**config)
        except ValidationError as e:
            print(e.json())
