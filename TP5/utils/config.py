from typing import Optional, List

import yaml
from pydantic import BaseModel, ValidationError, validator

DEFAULT_MAX_ITER = 10
DEFAULT_MIN_ERROR = 0.01
DEFAULT_LATENT_LAYER = 0.01
DEFAULT_INTERMEDIATE_LAYER = [25, 15]


class Config(BaseModel):
    font: Optional[int]
    selection_amount: Optional[int]
    max_iter: int = DEFAULT_MAX_ITER
    min_error: float = DEFAULT_MIN_ERROR
    latent_layer: int = DEFAULT_LATENT_LAYER
    intermediate_layers: List[int] = DEFAULT_INTERMEDIATE_LAYER

    @classmethod
    @validator('font')
    def must_be_valid_font(cls, font):
        if font is not None:
            if font not in [1, 2, 3]:
                raise ValueError('must be 1, 2 or 3')
        return font


def get_config(config_file: str) -> Config:
    with open(config_file) as cf:
        config = yaml.safe_load(cf)["config"]
        try:
            return Config(**config)
        except ValidationError as e:
            print(e.json())
