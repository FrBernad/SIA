from typing import Optional

import yaml
from pydantic import BaseModel, ValidationError

DEFAULT_RADIUS = 1
DEFAULT_K = 5
DEFAULT_MAX_ITER = 500


class KohonenConfig(BaseModel):
    radius: int = DEFAULT_RADIUS
    k: int = DEFAULT_K
    max_iter: int = DEFAULT_MAX_ITER


class Config(BaseModel):
    kohonen: KohonenConfig
    input_file: Optional[str]


def get_config(config_file: str) -> Config:
    with open(config_file) as cf:
        config = yaml.safe_load(cf)["config"]
        try:
            return Config(**config)
        except ValidationError as e:
            print(e.json())
