from pydantic import BaseModel, Field
from enum import Enum


class FieldTypes(Enum):
    """
    Enum for data types
    """

    STRING = "str"
    INTEGER = "int"
    FLOAT = "flt"
    BOOLEAN = "bln"
    RELATIONSHIP = "rel"
    ENUM = "enm"

    def to_description(self):
        if self == FieldTypes.STRING:
            return "A string value representing text."
        elif self == FieldTypes.INTEGER:
            return "An integer value representing a number."
        elif self == FieldTypes.FLOAT:
            return "A float value representing a floating point number."
        elif self == FieldTypes.BOOLEAN:
            return "A boolean value representing true or false."
        elif self == FieldTypes.RELATIONSHIP:
            return (
                "A relationship value representing a connection between two entities."
            )
        elif self == FieldTypes.ENUM:
            return "An enum value representing a set of predefined values."


class GeneratedField(BaseModel):
    """
    Data field to be used in a schema
    """

    name: str = Field(title="Name of the field")
    type: FieldTypes = Field(title="Type of the field")
    description: str = Field(title="Description of the field")
    many: bool = Field(default=False, title="Is the field a list?")
