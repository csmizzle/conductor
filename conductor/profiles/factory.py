"""
Pydantic models for the profiles module.
"""
from typing import Any, Type, MutableMapping, Union
from pydantic import Field
from conductor.flow.rag import CitedValueWithCredibility, WebSearchRAG
from conductor.flow.signatures import ExtractValue
from conductor.flow.specify import specify_description
from conductor.flow.models import NotAvailable
from pydantic import BaseModel
import dspy
from elasticsearch import Elasticsearch
from langchain_core.embeddings import Embeddings


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


def create_extract_value_with_custom_type(
    value_type: Type, value_description: str
) -> Type[ExtractValue]:
    """
    Create a specialized version of `ExtractValue` with a specific type for the `value` field.

    Args:
        value_type (Type): The type of the `value` field (e.g., str, int, float).
        value_description (str): The description for the `value` field.

    Returns:
        Type[ExtractValue]: A specialized version of `ExtractValue`.
    """

    # Create a new subclass with the updated description for the value field
    class CustomExtractValue(ExtractValue):  # Use generic specialization
        # this is crazy but it works
        value: Union[value_type, NotAvailable] = dspy.OutputField(
            desc=value_description
        )  # type: ignore

    return CustomExtractValue


def create_web_search_value_rag(
    return_class: Type,
    extract_value: Type[dspy.Signature],
) -> Type[WebSearchRAG]:
    """
    Factory function to create a custom subclass of WebSearchValueRAG with a modified forward method.

    Args:
        return_class (Type): The custom return class for the `forward` method.

    Returns:
        Type[WebSearchValueRAG]: A dynamically generated subclass of WebSearchValueRAG.
    """

    class CustomWebSearchValueRAG(WebSearchRAG):
        def __init__(self, elastic_id_retriever):
            super().__init__(elastic_id_retriever=elastic_id_retriever)
            self.generate_value = dspy.Predict(extract_value)

        def forward(self, question: str):
            # Run the forward method of the parent class
            cited_answer = super().forward(question=question)
            # Convert the answer to a value
            value = self.generate_value(question=question, answer=cited_answer.answer)
            # Return an instance of the specified return class
            return return_class(
                question=question,
                value=value.value,
                documents=cited_answer.documents,
                citations=cited_answer.citations,
                faithfulness=cited_answer.faithfulness,
                factual_correctness=cited_answer.factual_correctness,
                confidence=cited_answer.confidence,
                value_reasoning=cited_answer.answer_reasoning,
                source_credibility=cited_answer.source_credibility,
                source_credibility_reasoning=cited_answer.source_credibility_reasoning,
            )

    return CustomWebSearchValueRAG


def create_value_rag_pipeline(
    value_map: dict[str, tuple[Type[Any], str]],  # name, type, description
    elasticsearch: Elasticsearch,
    index_name: str,
    embeddings: Embeddings,
    cohere_api_key: str,
) -> dict[str, tuple[Type[WebSearchRAG], str]]:  # name, search class, description
    """
    Create a pipeline of WebSearchValueRAG classes based on the provided value map.

    Args:
        value_map (dict): A mapping of field names to (value_type, value_description) tuples.

    Returns:
        dict: A mapping of field names to (WebSearchValueRAG subclass, value_description) tuples.
    """
    rag_pipeline = {}
    for field_name, (value_type, value_description) in value_map.items():
        if not isinstance(value_type, MutableMapping):
            # Create a custom CitedValueWithCredibility subclass
            custom_cited_value = create_custom_cited_value(
                value_type=value_type, value_description=value_description
            )
            # Create a custom ExtractValue subclass
            custom_extract_value = create_extract_value_with_custom_type(
                value_type=value_type, value_description=value_description
            )
            # Create a custom WebSearchValueRAG subclass
            custom_rag = create_web_search_value_rag(
                return_class=custom_cited_value, extract_value=custom_extract_value
            )
            # Initialize the custom WebSearchValueRAG subclass
            custom_rag = custom_rag.with_elasticsearch_id_retriever(
                elasticsearch=elasticsearch,
                index_name=index_name,
                embeddings=embeddings,
                cohere_api_key=cohere_api_key,
            )
            rag_pipeline[field_name] = (custom_rag, value_description)
        # handle nested value maps
        else:
            nested_rag_pipeline = create_value_rag_pipeline(
                value_map=value_type,
                elasticsearch=elasticsearch,
                index_name=index_name,
                embeddings=embeddings,
                cohere_api_key=cohere_api_key,
            )
            rag_pipeline[field_name] = (nested_rag_pipeline, value_description)
    return rag_pipeline


def run_specified_rag(
    specification: str, name: str, description: str, rag: Type[WebSearchRAG]
):
    query = specify_description(
        name=name, description=description, specification=specification
    )
    return rag(question=query)


def run_value_rag_pipeline(
    specification: str,
    pipeline: dict[str, tuple[Type[WebSearchRAG], str]],
) -> dict[str, Type[CitedValueWithCredibility]]:
    """
    Run a pipeline of WebSearchValueRAG classes based on the provided value map.

    Args:
        rag_pipeline (dict): A mapping of field names to (WebSearchValueRAG subclass, value_description) tuples.

    Returns:
        dict: A mapping of field names to CitedValueWithCredibility instances.
    """
    values = {}
    for field_name, (rag, value_description) in pipeline.items():
        if not isinstance(rag, MutableMapping):
            value = run_specified_rag(
                specification=specification,
                name=field_name,
                description=value_description,
                rag=rag,
            )
            values[field_name] = value
        else:
            nested_values = run_value_rag_pipeline(
                specification=specification, pipeline=rag
            )
            values[field_name] = nested_values
    return values
