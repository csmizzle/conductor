"""
Signatures for template utilities.
"""
from conductor.template import models
import dspy


class GeneratedFieldsSignature(dspy.Signature):
    """
    Based on a prompt, generate a list of fields that will be used to define a data schema used for the data collection task downstream.
    The data schema should fit nicely into a standard SQL database schema.
    Avoid creating fields that are too specific to the data collection task.
    If a field represents many other entities connected to another entity, it makes sense to use a relationship.
    If a field can have set values, it may make sense to create an enum field.
    """

    prompt: str = dspy.InputField(description="Prompt to generate fields from")
    generated_fields: list[models.GeneratedField] = dspy.OutputField(
        description="Generated fields"
    )


class GeneratedEnumSignature(dspy.Signature):
    """
    Create a set of enum values for a field based on the name and description of the field.
    Be sure that the enum values are relevant to the field and provide a sensible range of options for the field.
    The enum values should be a list of strings that can be easily transformed into a set of values for a field.
    Do not create enum values that would be valid Python Identifiers if the they were upper case and spaces were replaced with underscores.
    Do not use any special characters in the enum values, only letters, numbers, and spaces.
    """

    name: str = dspy.InputField(description="Name of the enum field")
    description: str = dspy.InputField(description="Description of the enum field")
    generated_values: list[str] = dspy.OutputField(
        description="Values of the enum field"
    )


class GeneratedRelationshipSignature(dspy.Signature):
    """
    Based on a field name and description generate a list of fields that will be used to define a data schema used for the data collection task downstream.
    The data schema should fit nicely into a standard SQL database schema.
    Do not create relationship type fields.
    Avoid creating fields that are too specific to the data collection task.
    If a field can have set values, it may make sense to create an enum field.
    """

    name: str = dspy.InputField(description="Name of the relationship")
    description: str = dspy.InputField(description="Description of the relationship")
    generated_fields: list[models.GeneratedField] = dspy.OutputField(
        description="Generated fields"
    )


class GeneratedSchemaNameSignature(dspy.Signature):
    """
    Based on a prompt, generate a name for a template that will house a set of fields.
    The name should represent the single entity the template will be used to collect data on.
    For example, if the prompt is "I am conducting due diligence on a company", the template name could be "Company".
    If the prompt was "I am researching pitcher performance in baseball", the template name could be "Pitcher".
    """

    prompt = dspy.InputField(description="Prompt to generate template name from")
    generated_schema_name = dspy.OutputField(description="Generated schema name")


class GeneratedRelationshipSchemaNameSignature(dspy.Signature):
    """
    Based on a name and description, generate a name for a template that will house a set of fields.
    The name should be short, consistent, and descriptive.
    Limit the name to 3 words or less.
    """

    name: str = dspy.InputField(description="Name of the relationship")
    description: str = dspy.InputField(description="Description of the relationship")
    generated_schema_name = dspy.OutputField(description="Generated template name")


class GeneratedResearchQuestionsSignature(dspy.Signature):
    """
    Based on a prompt, and the subsequent generated fields, generate a list of research questions that provide additional second level analysis on the data that pair nicely with the data schema and prompt.
    The research questions should be open-ended and should not be leading.
    The research questions should be relevant to the prompt, augment and provide additional context to the data schema, and provide a basis for further analysis.
    Use the "n" parameter to specify the number of research questions to generate.
    """

    prompt: str = dspy.InputField(
        description="Prompt to generate research questions from"
    )
    generated_fields: list[models.GeneratedField] = dspy.InputField(
        description="Generated fields"
    )
    n: int = dspy.InputField(
        default=5, description="Number of research questions -to generate"
    )
    generated_research_questions: list[str] = dspy.OutputField(
        description="Generated research questions"
    )
