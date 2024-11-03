import json
from pydantic import BaseModel
from typing import Union


def save_model_to_test_data(
    model: Union[list[BaseModel], BaseModel], filename: str
) -> None:
    if isinstance(model, list):
        with open(f"./tests/data/{filename}", "w") as f:
            json.dump([m.model_dump() for m in model], f, indent=4)
    else:
        with open(f"./tests/data/{filename}", "w") as f:
            json.dump(model.model_dump(), f, indent=4)


def load_model_from_test_data(model: BaseModel, filename: str) -> BaseModel:
    with open(f"./tests/data/{filename}", "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [model.model_validate(m) for m in data]
    else:
        return model.model_validate(data)
