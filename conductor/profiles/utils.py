"""
Profiles should be decomposed into a series of questions based on field descriptions.
This should be executed and then the results should a value that is inserted into a profile.
We can do this through reading the field descriptions and then generating a series of questions.
"""
from pydantic import BaseModel, InstanceOf
from conductor.flow.specify import specify_description


def get_model_descriptions(model: InstanceOf[BaseModel]) -> dict[str, str]:
    """
    Get the descriptions of the fields in a model
    """
    description = {}
    properties = model.model_json_schema()["properties"]
    for key in properties:
        description[key] = properties[key]["description"]
    return description


def specify_model(model: InstanceOf[BaseModel], specification: str) -> dict[str, str]:
    """
    Specify a model
    """
    descriptions = get_model_descriptions(model=model)
    specified_fields = {}
    for field in descriptions:
        specified_fields[field] = specify_description(
            name=field, description=descriptions[field], specification=specification
        )
    return specified_fields
