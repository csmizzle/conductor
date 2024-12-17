from conductor.template import signatures
from conductor.template import models
import dspy
from loguru import logger


def generate_template_schema(prompt: str) -> dspy.Prediction:
    """
    Generate a list of fields that will be used to define a data schema used for the data collection task downstream.

    Args:
    - prompt: Prompt to generate fields from

    Returns:
    - Generated fields
    """
    template_generator = dspy.ChainOfThought(signatures.GeneratedFieldsSignature)
    return template_generator(prompt=prompt)


class SchemaGenerator:
    def __init__(
        self,
        prompt: str,
        generate_nested_enums: bool,
        generate_nested_relationship: bool,
    ) -> None:
        self.prompt = prompt
        self.generate_enum = dspy.ChainOfThought(signatures.GeneratedEnumSignature)
        self.generate_relationship = dspy.ChainOfThought(
            signatures.GeneratedRelationshipSignature
        )
        self.generate_nested_enums = generate_nested_enums
        self.generate_nested_relationship = generate_nested_relationship

    def _unpack_enum_values(
        self, generated_enums: list[str], many: bool
    ) -> tuple[list[str], str]:
        return (generated_enums, "many" if many else "single")

    def _unpack_generated_fields(
        self, generated_fields: list[models.GeneratedField]
    ) -> dict:
        values = {}
        for field in generated_fields:
            if field.type == field.type.ENUM:
                if self.generate_nested_enums:
                    logger.info(f"Generating enums for field: {field.name}")
                    generated_enums = self.generate_enum(
                        name=field.name, description=field.description
                    ).generated_values
                    values[field.name] = (
                        self._unpack_enum_values(generated_enums, field.many),
                        field.description,
                    )
            else:
                values[field.name] = (field.type.value, field.description)
        return values

    def generate(self) -> dict:
        """
        Generate a list of fields that will be used to define a data schema used for the data collection task downstream.

        Returns:
        - Generated fields
        """
        value_map = {}
        generated_fields = generate_template_schema(self.prompt)
        for field in generated_fields.generated_fields:
            if field.type in [
                models.FieldTypes.STRING,
                models.FieldTypes.BOOLEAN,
                models.FieldTypes.INTEGER,
                models.FieldTypes.FLOAT,
            ]:
                value_map[field.name] = (field.type.value, field.description)
            if field.type == models.FieldTypes.RELATIONSHIP:
                if self.generate_nested_relationship:
                    logger.info(f"Generating relationship for field: {field.name}")
                    generated_relationships = self.generate_relationship(
                        name=field.name, description=field.description
                    ).generated_fields
                    value_map[field.name] = (
                        self._unpack_generated_fields(generated_relationships),
                        field.description,
                    )
                else:
                    value_map[field.name] = ({}, field.description)
            if field.type == field.type.ENUM:
                if self.generate_nested_enums:
                    logger.info(f"Generating enums for field: {field.name}")
                    generated_enums = self.generate_enum(
                        name=field.name, description=field.description
                    ).generated_values
                    value_map[field.name] = (
                        self._unpack_enum_values(generated_enums, field.many),
                        field.description,
                    )
                else:
                    value_map[field.name] = ([], field.description)
        return value_map
