from conductor.template import signatures
from conductor.template import models
import dspy
from loguru import logger


class SchemaGenerator:
    def __init__(
        self,
        prompt: str,
        generate_nested_enums: bool = False,
        generate_nested_relationship: bool = False,
    ) -> None:
        self.prompt = prompt
        self.generate_template = dspy.ChainOfThought(
            signatures.GeneratedFieldsSignature
        )
        self.generate_enum = dspy.ChainOfThought(signatures.GeneratedEnumSignature)
        self.generate_relationship = dspy.ChainOfThought(
            signatures.GeneratedRelationshipSignature
        )
        self.generate_nested_enums = generate_nested_enums
        self.generate_nested_relationship = generate_nested_relationship
        self.generate_schema_name = dspy.ChainOfThought(
            signatures.GeneratedSchemaNameSignature
        )

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

    def generate(self) -> models.ValueMap:
        """
        Generate a list of fields that will be used to define a data schema used for the data collection task downstream.

        Returns:
        - Generated fields
        """
        fields = []
        generated_fields = self.generate_template(prompt=self.prompt)
        for field in generated_fields.generated_fields:
            if field.type in [
                models.FieldTypes.STRING,
                models.FieldTypes.BOOLEAN,
                models.FieldTypes.INTEGER,
                models.FieldTypes.FLOAT,
            ]:
                fields.append(
                    models.ValueField(
                        name=field.name, field=(field.type.value, field.description)
                    )
                )
            if field.type == models.FieldTypes.RELATIONSHIP:
                if self.generate_nested_relationship:
                    logger.info(f"Generating relationship for field: {field.name}")
                    generated_relationships = self.generate_relationship(
                        name=field.name, description=field.description
                    ).generated_fields
                    fields.append(
                        models.ValueRelationshipField(
                            name=field.name,
                            field=(
                                self._unpack_generated_fields(generated_relationships),
                                field.description,
                            ),
                        )
                    )
                else:
                    fields.append(
                        models.ValueRelationshipField(
                            name=field.name, field=({}, field.description)
                        )
                    )
            if field.type == field.type.ENUM:
                if self.generate_nested_enums:
                    logger.info(f"Generating enums for field: {field.name}")
                    generated_enums = self.generate_enum(
                        name=field.name, description=field.description
                    ).generated_values
                    fields.append(
                        models.ValueEnumField(
                            name=field.name,
                            field=(
                                self._unpack_enum_values(generated_enums, field.many),
                                field.description,
                            ),
                        )
                    )
                else:
                    fields.append(
                        models.ValueEnumField(
                            name=field.name, field=(([], ""), field.description)
                        )
                    )
        # generate schema name
        schema_name = self.generate_schema_name(
            prompt=self.prompt
        ).generated_schema_name
        return models.ValueMap(name=schema_name, fields=fields)
