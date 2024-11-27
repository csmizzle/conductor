"""
Pydantic models for the profiles module.
"""
from typing import Any, Type
from pydantic import Field
from conductor.flow.rag import CitedValueWithCredibility
from pydantic import BaseModel


def create_custom_cited_value(
    value_type: Type[Any], value_description: str
) -> Type[CitedValueWithCredibility]:
    """
    Dynamically creates a subclass of `CitedValueWithCredibility` with a modified `value` field.

    Args:
        value_type (Type[Any]): The new type for the `value` field.
        value_description (str): The new description for the `value` field.

    Returns:
        Type[CitedValueWithCredibility]: A dynamically generated subclass.
    """
    # Dynamically create a subclass of CitedValueWithCredibility
    model = type(
        "CustomCitedValueWithCredibility",
        (CitedValueWithCredibility,),
        {
            "__annotations__": {
                **CitedValueWithCredibility.__annotations__,
                "value": value_type,  # Update the type of the `value` field
            },
            "value": Field(description=value_description),  # Update the field metadata
        },
    )
    return model


def create_custom_cited_model(
    model_name: str, value_map: dict[str, tuple[Type[Any], str]]
) -> Type[BaseModel]:
    """
    Dynamically creates a subclass of `BaseModel` with custom `CitedValueWithCredibility` fields.

    Args:
        model_name (str): The name of the new model.
        value_map (dict): A mapping of field names to (value_type, value_description) tuples.

    Returns:
        Type[BaseModel]: A dynamically generated subclass.
    """
    # Dynamically create a subclass of BaseModel
    model = type(
        model_name,
        (BaseModel,),
        {
            "__annotations__": {
                field_name: (create_custom_cited_value(value_type, value_description))
                for field_name, (value_type, value_description) in value_map.items()
            },
        },
    )
    return model
