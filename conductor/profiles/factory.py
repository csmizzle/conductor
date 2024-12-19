"""
Factories for creating custom classes and pipelines for Evrim Profiles.
"""
from typing import Any, Type, Union, List, Dict, Optional, Tuple
from pydantic import Field, create_model, InstanceOf
from conductor.flow.rag import CitedValueWithCredibility, WebSearchRAG
from conductor.flow.signatures import ExtractValue
from conductor.flow.specify import specify_description
from conductor.graph import models as graph_models
from conductor.graph import signatures as graph_signatures
from conductor.graph.extraction import RelationshipRAGExtractor
from conductor.graph.create import create_deduplicated_graph
from conductor.flow.rag import WebDocumentRetriever
from conductor.rag.embeddings import BedrockEmbeddings
from conductor.flow.retriever import ElasticDocumentIdRMClient
from conductor.profiles.signatures import (
    generate_relationship,
    generate_relationship_specification,
    evaluate_extraction,
)
from elasticsearch import Elasticsearch
from pydantic import BaseModel
from loguru import logger
import dspy
from langchain_core.embeddings import Embeddings
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Subclassing to add a dynamic field
def create_subclass_with_dynamic_fields(
    model_name: str,
    base_class: Type[BaseModel],
    new_fields: Dict[
        str, Tuple[Any, Optional[Any], str]
    ],  # name, type, default, description
) -> Type[BaseModel]:
    fields = {
        field_name: (
            field_type,
            Field(default=default_value, description=field_description),
        )
        for field_name, (
            field_type,
            default_value,
            field_description,
        ) in new_fields.items()
    }
    return create_model(
        model_name,
        **fields,
        __base__=base_class,
    )


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
    if value_type is bool:
        # Create a new subclass with the updated description for the value field, with boolean types we want to return false not null
        class CustomExtractValue(ExtractValue):
            """
            Distill an answer to an answer into a value that would fit into a database.
            Use the question to help understand which value to extract.
            If there isn't a value return the None value.
            """

            # this is crazy but it works
            value: value_type = dspy.OutputField(desc=value_description)  # type: ignore

        return CustomExtractValue
    else:
        # Create a new subclass with the updated description for the value field with a Union for Null value
        class CustomExtractValue(ExtractValue):
            """
            Distill an answer to an answer into a value that would fit into a database.
            Use the question to help understand which value to extract.
            If there isn't a value return the None value.
            """

            value: Union[value_type, None] = dspy.OutputField(desc=value_description)  # type: ignore
            # this is crazy but it works

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
            value = self.generate_value(
                question=question, answer=cited_answer.answer
            ).value
            # Return an instance of the specified return class
            return return_class(
                question=question,
                value=value,
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


def run_specified_rag(
    specification: str, name: str, description: str, rag: Type[WebSearchRAG]
):
    query = specify_description(
        name=name, description=description, specification=specification
    )
    return rag(question=query)


def enum_factory(
    name: str, values: list[tuple[str, Any]], not_available: bool = False
) -> Enum:
    """
    Creates an Enum dynamically.

    Args:
        name (str): The name of the Enum class.
        values (list[tuple[str, Any]]): A list of tuples where the first element is the name of the Enum member,
                                        and the second is its value.

    Returns:
        Enum: A dynamically created Enum class.
    """
    if not name.isidentifier():
        raise ValueError(
            f"The provided name '{name}' is not a valid Python identifier."
        )

    if any(not isinstance(v, tuple) or len(v) != 2 for v in values):
        raise ValueError("Values must be a list of tuples (name, value).")

    # Validate names
    for item_name, _ in values:
        if not item_name.isidentifier():
            raise ValueError(
                f"Enum member name '{item_name}' is not a valid Python identifier."
            )
    if not_available:
        values.append(("NOT_AVAILABLE", "NOT_AVAILABLE"))

    # Dynamically create and return the Enum
    return Enum(name, {item_name: item_value for item_name, item_value in values})


class GraphModelFactoryPipeline:
    """
    Create the aspects of graph models dynamically to be used in Evrim Profiles.
    """

    def __init__(
        self,
        triple_types: dict[
            tuple[str, str]
        ],  # relationship_type, source entity type, target entity type
    ) -> None:
        self.triple_types = triple_types
        # Create the entity and relationship types
        self.relationship_type: Enum = None
        self.entity_type: Enum = None
        # Create the entity and relationship models
        self.entity_model: Type[graph_models.Entity] = None
        self.relationship_model: Type[graph_models.Relationship] = None
        # create cited relationship
        self.cited_relationship: Type[
            graph_models.CitedRelationshipWithCredibility
        ] = None
        # create aggregated models
        self.aggregated_cited_entity: Type[graph_models.AggregatedCitedEntity] = None
        self.aggregated_cited_relationship: Type[
            graph_models.AggregatedCitedRelationship
        ] = None
        self.aggregated_cited_graph: Type[graph_models.AggregatedCitedGraph] = None
        # create the triple types to execute the pipeline
        self.created_triple_type_model: Type[graph_models.TripleTypes] = None
        # triple type inputs
        self.triple_type_inputs: list[graph_models.TripleType] = []

    @staticmethod
    def _normalize_string(string: str) -> str:
        """
        Normalize a string by removing spaces and converting it to uppercase.

        Args:
            string (str): The string to normalize.

        Returns:
            str: The normalized string.
        """
        return string.upper().replace(" ", "_")

    def _create_relationship_types(self) -> None:
        """
        Create the RelationshipType Enum dynamically by using the keys of the triple_types dictionary.
        """
        relationship_values = [
            (self._normalize_string(key), self._normalize_string(key))
            for key in self.triple_types.keys()
        ]
        self.relationship_type = enum_factory(
            "CustomRelationshipType", relationship_values
        )

    def _create_entity_types(self) -> None:
        """
        Create the EntityType Enum dynamically by using the unique values of the triple_types dictionary.
        """
        entity_values = set(
            [entity for entities in self.triple_types.values() for entity in entities]
        )
        entity_values = [
            (self._normalize_string(value), self._normalize_string(value))
            for value in entity_values
        ]
        self.entity_type = enum_factory("CustomEntityType", entity_values)

    def _create_entity_base_model(self) -> None:
        """
        Create the Entity BaseModel dynamically with the EntityType Enum.
        """
        self.entity_model = create_subclass_with_dynamic_fields(
            model_name="CustomEntity",
            base_class=graph_models.Entity,
            new_fields={
                "entity_type": (self.entity_type, None, "Entity type"),  # type: ignore
            },
        )

    def _create_relationship_base_model(self) -> None:
        """
        Create the Relationship BaseModel dynamically with the RelationshipType Enum.
        """
        # Dynamically create a subclass of CitedValueWithCredibility
        self.relationship_model = create_subclass_with_dynamic_fields(
            model_name="CustomRelationship",
            base_class=graph_models.Relationship,
            new_fields={
                "source": (self.entity_model, None, "Source entity"),
                "target": (self.entity_model, None, "Target entity"),
                "relationship_type": (
                    self.relationship_type,
                    None,
                    "Relationship type",
                ),
            },
        )

    def _create_cited_relationship(self) -> None:
        """
        Create the CitedRelationship BaseModel dynamically with the Relationship BaseModel.
        """
        self.created_cited_relationship = create_subclass_with_dynamic_fields(
            model_name="CustomCitedRelationshipWithCredibility",
            base_class=graph_models.CitedRelationshipWithCredibility,
            new_fields={
                "source": (self.entity_model, None, "Source entity"),
                "target": (self.entity_model, None, "Target entity"),
            },
        )

    def _create_aggregated_cited_entity(self) -> None:
        """
        Create the CitedEntity BaseModel dynamically with the Entity BaseModel.
        """
        self.cited_entity = create_subclass_with_dynamic_fields(
            model_name="CustomCitedEntity",
            base_class=graph_models.AggregatedCitedEntity,
            new_fields={
                "entity": (self.entity_model, None, "The entity"),
            },
        )

    def _create_aggregated_cited_relationship(self) -> None:
        """
        Create the CitedRelationship BaseModel dynamically with the Relationship BaseModel.
        """
        self.cited_relationship = create_subclass_with_dynamic_fields(
            model_name="CustomCitedRelationship",
            base_class=graph_models.AggregatedCitedRelationship,
            new_fields={
                "source": (self.entity_model, None, "Source entity"),
                "target": (self.entity_model, None, "Target entity"),
            },
        )

    def _create_aggregated_cited_graph(self) -> None:
        """
        Create an aggregated cited graph model dynamically with the CitedEntity and CitedRelationship BaseModels.
        """
        self.aggregated_cited_graph = create_subclass_with_dynamic_fields(
            model_name="CustomAggregatedCitedGraph",
            base_class=graph_models.AggregatedCitedGraph,
            new_fields={
                "entities": (List[self.cited_entity], None, "List of cited entities"),
                "relationships": (
                    List[self.cited_relationship],
                    None,
                    "List of cited relationships",
                ),
            },
        )

    def _create_triple_type_model(self) -> None:
        """
        Iterate over created entity and relationship types to create the triple types enum.
        """
        self.created_triple_type_model = create_subclass_with_dynamic_fields(
            model_name="CustomTripleTypes",
            base_class=graph_models.TripleType,
            new_fields={
                "relationship_type": (
                    self.relationship_type,
                    None,
                    "Relationship type",
                ),
                "source": (self.entity_type, None, "Source entity type"),
                "target": (self.entity_type, None, "Target entity type"),
            },
        )

    def create_models(self) -> None:
        """
        Create the entity and relationship models dynamically.
        """
        self._create_relationship_types()
        self._create_entity_types()
        self._create_entity_base_model()
        self._create_relationship_base_model()
        self._create_cited_relationship()
        self._create_aggregated_cited_entity()
        self._create_aggregated_cited_relationship()
        self._create_aggregated_cited_graph()
        self._create_triple_type_model()

    def create_triple_type_input(self) -> None:
        """
        Iterate over normalized triple types to create the input for the triple type model.
        """
        for relationship_type, (
            source_entity_type,
            target_entity_type,
        ) in self.triple_types.items():
            relationship_type = self._normalize_string(relationship_type)
            source_entity_type = self._normalize_string(source_entity_type)
            target_entity_type = self._normalize_string(target_entity_type)
            triple_type = self.created_triple_type_model(
                relationship_type=self.relationship_type[relationship_type],
                source=self.entity_type[source_entity_type],
                target=self.entity_type[target_entity_type],
            )
            self.triple_type_inputs.append(triple_type)


class GraphSignatureFactory:
    def __init__(
        self,
        triple_type_model: Type[graph_models.TripleType],
        relationship: Type[graph_models.Relationship],
    ) -> None:
        self.triple_type_model = triple_type_model
        self.relationship = relationship
        self.relationship_query: Type[graph_signatures.RelationshipQuery] = None
        self.extracted_relationships: Type[
            graph_signatures.ExtractedRelationships
        ] = None
        self.relationship_reasoning: Type[graph_signatures.RelationshipReasoning] = None

    def create_relationship_query(self) -> None:
        """
        Class factory for RelationshipQuery signature.
        """

        class CustomRelationshipQuery(graph_signatures.RelationshipQuery):
            triple_type: self.triple_type_model = dspy.InputField(
                description="Triple containing source_type, relationship_type, and target_type",
                prefix="Triple: ",
            )

        self.relationship_query = CustomRelationshipQuery

    def create_extracted_relationships(self) -> None:
        """
        Class factory for ExtractedRelationships signature.
        """

        class CustomExtractedRelationships(graph_signatures.ExtractedRelationships):
            triple_type: self.triple_type_model = dspy.InputField(
                description="Triple containing source_type, relationship_type, and target_type",
                prefix="Triple: ",
            )
            relationships: list[self.relationship] = dspy.OutputField(
                description="Extracted relationships", prefix="Relationships: "
            )

        self.extracted_relationships = CustomExtractedRelationships

    def create_relationship_reasoning(self) -> None:
        """
        Class factory for RelationshipReasoning signature.
        """

        class CustomRelationshipReasoning(graph_signatures.RelationshipReasoning):
            relationship: self.relationship = dspy.InputField(
                description="Relationships to reason about", prefix="Relationships: "
            )

        self.relationship_reasoning = CustomRelationshipReasoning

    def create_signatures(self) -> None:
        """
        Create the relationship query, extracted relationships, and relationship reasoning signatures.
        """
        self.create_relationship_query()
        self.create_extracted_relationships()
        self.create_relationship_reasoning()

    def create_signatures_map(self) -> dict:
        """
        Create a mapping of signature names to their corresponding classes.

        Returns:
            dict: A mapping of signature names to classes.
        """
        return {
            "create_relationship_query": self.relationship_query,
            "extract_relationships": self.extracted_relationships,
            "create_relationship_reasoning": self.relationship_reasoning,
        }


class RelationshipRAGExtractorFactory:
    """
    Factory to dynamically configure and instantiate RelationshipRAGExtractor
    """

    def __init__(
        self,
        rag_instance: Type[dspy.Module],
        signature_map: Dict[str, dspy.Signature],
        specification: str,
        triple_types: List[graph_models.TripleType],
        cited_relationship_model: Type[graph_models.CitedRelationshipWithCredibility],
    ) -> None:
        """
        Initializes the factory with reusable components.

        :param rag_instance: Instance of the RAG module for document retrieval.
        :param signature_map: A mapping of method names to their signature classes.
        """
        self.rag_instance = rag_instance
        self.signature_map = signature_map
        self.cited_relationship_model = cited_relationship_model
        self.specification = specification
        self.triple_types = triple_types
        self.created_rag_instance: Type[RelationshipRAGExtractor] = None

    def create_instance(self) -> "RelationshipRAGExtractor":
        """
        Creates a configured instance of RelationshipRAGExtractor.

        :param specification: Specification string.
        :param triple_types: List of TripleType objects.
        :return: Configured RelationshipRAGExtractor instance.
        """
        create_relationship_query_signature = self.signature_map[
            "create_relationship_query"
        ]
        extracted_relationships_signature = self.signature_map["extract_relationships"]
        create_relationship_reasoning_signature = self.signature_map[
            "create_relationship_reasoning"
        ]

        class ConfiguredRelationshipRAGExtractor(RelationshipRAGExtractor):
            def __init__(
                self,
                specification: str,
                triple_types: List[graph_models.TripleType],
                rag: Type[dspy.Module],
                cited_relationship_model: Type[
                    graph_models.CitedRelationshipWithCredibility
                ],
            ) -> None:
                super().__init__(
                    specification=specification,
                    triple_types=triple_types,
                    rag=rag,
                    cited_relationship_model=cited_relationship_model,
                )
                self.created_rag_instance: Type[RelationshipRAGExtractor] = None
                self.create_relationship_query = dspy.ChainOfThought(
                    create_relationship_query_signature
                )
                self.extract_relationships = dspy.ChainOfThought(
                    extracted_relationships_signature
                )
                self.create_relationship_reasoning = dspy.ChainOfThought(
                    create_relationship_reasoning_signature
                )

        # Instantiate and return the configured extractor
        self.created_rag_instance = ConfiguredRelationshipRAGExtractor(
            specification=self.specification,
            triple_types=self.triple_types,
            rag=self.rag_instance,
            cited_relationship_model=self.cited_relationship_model,
        )


def create_dynamic_relationship_extraction(
    specification: str,
    triple_types: dict[
        str, tuple[str, str]
    ],  # relationship_type, source entity type, target entity type
    elasticsearch: Elasticsearch,
    index_name: str,
) -> InstanceOf[RelationshipRAGExtractor]:
    model_factory = GraphModelFactoryPipeline(triple_types)
    model_factory.create_models()
    model_factory.create_triple_type_input()
    signature_factory = GraphSignatureFactory(
        triple_type_model=model_factory.created_triple_type_model,
        relationship=model_factory.relationship_model,
    )
    signature_factory.create_signatures()
    signature_map = signature_factory.create_signatures_map()
    retriever = ElasticDocumentIdRMClient(
        elasticsearch=elasticsearch,
        index_name=index_name,
        embeddings=BedrockEmbeddings(),
    )
    rag = WebDocumentRetriever(elastic_id_retriever=retriever)
    rag_factory = RelationshipRAGExtractorFactory(
        specification=specification,
        triple_types=model_factory.triple_type_inputs,
        rag_instance=rag,
        signature_map=signature_map,
        cited_relationship_model=model_factory.created_cited_relationship,
    )
    rag_factory.create_instance()
    return rag_factory.created_rag_instance


def build_value_rag_pipeline(
    value_type: Type[Any],
    value_description: str,
    elasticsearch: Elasticsearch,
    index_name: str,
    embeddings: Embeddings,
    cohere_api_key: str = None,
) -> InstanceOf[WebSearchRAG]:
    if value_type is not bool:
        custom_cited_value = create_subclass_with_dynamic_fields(
            model_name="CustomCitedValueWithCredibility",
            base_class=CitedValueWithCredibility,
            new_fields={"value": (Union[value_type, None], None, value_description)},
        )
    else:
        custom_cited_value = create_subclass_with_dynamic_fields(
            model_name="CustomCitedValueWithCredibility",
            base_class=CitedValueWithCredibility,
            new_fields={"value": (value_type, None, value_description)},
        )
    custom_extract_value = create_extract_value_with_custom_type(
        value_type=value_type, value_description=value_description
    )
    custom_rag = create_web_search_value_rag(
        return_class=custom_cited_value, extract_value=custom_extract_value
    )
    custom_rag = custom_rag.with_elasticsearch_id_retriever(
        elasticsearch=elasticsearch,
        index_name=index_name,
        embeddings=embeddings,
        cohere_api_key=cohere_api_key,
    )
    return custom_rag


def create_value_rag_pipeline(
    value_map: dict[
        str, dict[str, tuple[Type[Any], str]]
    ],  # model_name [name, type, description
    elasticsearch: Elasticsearch,
    index_name: str,
    embeddings: Embeddings,
    cohere_api_key: str,
    add_not_available: bool = False,
) -> dict[str, tuple[Type[WebSearchRAG], str]]:  # name, search class, description
    """
    Create a pipeline of WebSearchValueRAG classes based on the provided value map.

    Args:
        value_map (dict): A mapping of field names to (value_type, value_description) tuples.

    Returns:
        dict: A mapping of field names to (WebSearchValueRAG subclass, value_description) tuples.
    """
    rag_pipeline = {}  # model_name [name, type, description]
    for model_name in value_map:
        rag_pipeline[model_name] = {}
        for field_name, (value_type, value_description) in value_map[
            model_name
        ].items():
            if value_type in [int, float, str, bool]:
                custom_rag = build_value_rag_pipeline(
                    value_type=value_type,
                    value_description=value_description,
                    elasticsearch=elasticsearch,
                    index_name=index_name,
                    embeddings=embeddings,
                    cohere_api_key=cohere_api_key,
                )
                rag_pipeline[model_name][field_name] = (custom_rag, value_description)
            # handle enum creation prior to building the pipeline
            elif isinstance(value_type, tuple):
                field_name = field_name.replace(" ", "_").lower()  # snake sane
                enum_name = field_name.title().replace(" ", "")  # camel case
                enum = enum_factory(
                    name=enum_name,
                    values=[
                        (
                            value.replace(" ", "_").upper(),
                            value.replace(" ", "_").upper(),
                        )
                        for value in value_type[0]
                    ],
                    not_available=add_not_available,
                )
                if value_type[1] == "many":
                    enum = List[enum]
                elif value_type[1] == "single":
                    enum = enum
                else:
                    raise ValueError(
                        f"Enum type must be 'single' or 'many', not {value_type[1]}"
                    )
                custom_rag = build_value_rag_pipeline(
                    value_type=enum,
                    value_description=value_description,
                    elasticsearch=elasticsearch,
                    index_name=index_name,
                    embeddings=embeddings,
                    cohere_api_key=cohere_api_key,
                )
                rag_pipeline[model_name][field_name] = (custom_rag, value_description)
            # handle nested value maps
            elif isinstance(value_type, dict):
                # create a triple type map (relationship_type, source entity type, target entity type)
                relationship = generate_relationship(
                    source=model_name,
                    target=field_name,
                )
                # create profile pipeline
                profile_pipeline = {}
                for _field_name, (
                    _value_type,
                    _value_description,
                ) in value_type.items():
                    # do terrible nested checking
                    if isinstance(_value_type, tuple):
                        field_name = _field_name.replace(" ", "_").lower()  # snake sane
                        enum_name = _field_name.title().replace(" ", "")  # camel case
                        enum = enum_factory(
                            name=enum_name,
                            values=[
                                (
                                    value.replace(" ", "_").upper(),
                                    value.replace(" ", "_").upper(),
                                )
                                for value in _value_type[0]
                            ],
                            not_available=add_not_available,
                        )
                        _value_type = enum
                    else:
                        _value_type = _value_type
                    custom_rag = build_value_rag_pipeline(
                        value_type=_value_type,
                        value_description=_value_description,
                        elasticsearch=elasticsearch,
                        index_name=index_name,
                        embeddings=embeddings,
                        cohere_api_key=cohere_api_key,
                    )
                    profile_pipeline[_field_name] = (custom_rag, _value_description)
                # map the two pipelines together
                profile_relationship_pipeline = {
                    "triple_type": {relationship: (model_name, field_name)},
                    "profile_pipeline": profile_pipeline,
                }
                # add the profile pipeline to the rag pipeline
                rag_pipeline[model_name][
                    field_name
                ] = profile_relationship_pipeline  # check length when executing to build pipeline dynamically
    rag_pipeline["_resources"] = {}
    # set up the pipeline resources for downstream access
    rag_pipeline["_resources"]["elasticsearch"] = elasticsearch
    rag_pipeline["_resources"]["index_name"] = index_name
    rag_pipeline["_resources"]["embeddings"] = embeddings
    rag_pipeline["_resources"]["cohere_api_key"] = cohere_api_key
    return rag_pipeline


def process_relationship(
    relationship: InstanceOf[graph_models.AggregatedCitedRelationship],
    profile_pipeline: dict,
    quality_threshold: int = 3,
    generate_profile: bool = True,
) -> dict:
    # Run the profile pipeline
    profile = {}
    extraction_evaluation = evaluate_extraction(
        source_name=relationship.source.name,
        source_type=relationship.source.entity_type,
        target_name=relationship.target.name,
        target_type=relationship.target.entity_type,
        relationship_type=relationship.relationship_type,
    )
    if extraction_evaluation.quality >= quality_threshold:
        if generate_profile:
            relationship_specification = generate_relationship_specification(
                source=relationship.source.name,
                relationship_type=relationship.relationship_type,
                target=relationship.target.name,
            )
            logger.info(f"Running profile pipeline for {relationship_specification}")
            for _field_name, _entry in profile_pipeline.items():
                profile[_field_name] = run_specified_rag(
                    specification=relationship_specification,
                    name=_field_name,
                    rag=_entry[0],  # WebSearchValueRAG subclass
                    description=_entry[1],  # value_description
                )
        else:
            logger.info("Skipping profile ...")
    else:
        logger.warning(
            f"Extraction evaluation quality is below threshold for {relationship.source.name}->{relationship.relationship_type}->{relationship.target.name}"
        )
        logger.warning(f"Quality: {extraction_evaluation.reasoning}")
    return {
        "relationship": relationship,
        "quality": {
            "value": extraction_evaluation.quality,
            "reasoning": extraction_evaluation.reasoning,
        },
        "profile": profile,
    }


def parallel_process_relationships(
    deduplicated_graph: graph_models.AggregatedCitedGraph,
    profile_pipeline: dict,
    max_workers: int = 4,
    quality_threshold: int = 3,
    generate_profiles: bool = True,
):
    profiled_relationships = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the thread pool
        future_to_relationship = {
            executor.submit(
                process_relationship,
                relationship,
                profile_pipeline,
                quality_threshold,
                generate_profiles,
            ): relationship
            for relationship in deduplicated_graph.relationships
        }
        for future in tqdm(
            as_completed(future_to_relationship), total=len(future_to_relationship)
        ):
            try:
                # Collect the processed relationship
                result = future.result()
                if result:
                    profiled_relationships.append(result)
            except Exception as e:
                logger.error(f"Error processing relationship: {e}")
    return profiled_relationships


def run_value_rag_pipeline(
    specification: str,
    pipeline: dict[str, tuple[Type[WebSearchRAG], str]],
    max_workers: int = 4,
    quality_threshold: int = 3,
    generate_relationship_profiles: bool = False,
) -> dict[str, list[dict[str, Type[CitedValueWithCredibility]]]]:
    """
    Run a pipeline of WebSearchValueRAG classes based on the provided value map.

    Args:
        specification (str): The specification string.
        pipeline (dict): A mapping of field names to (WebSearchValueRAG subclass, value_description) tuples.
        max_workers (int): The maximum number of workers to use for parallel processing.
        quality_threshold (int): The minimum quality threshold for extraction evaluation.
        generate_relationship_profiles (bool): Whether to generate profiles for relationships.

    Returns:
        dict: A mapping of field names to CitedValueWithCredibility instances.
    """
    resources = pipeline.pop("_resources")
    values = {}
    for model_name in pipeline:
        values[model_name] = {}
        for field_name, entry in pipeline[model_name].items():
            if isinstance(entry, tuple):  # handle single value maps
                value = run_specified_rag(
                    specification=specification,
                    name=field_name,
                    description=entry[1],  # value_description
                    rag=entry[0],  # WebSearchValueRAG subclass
                )
                values[model_name][field_name] = value
            if isinstance(
                entry, dict
            ):  # handle nested value maps with relationship extraction
                profile_pipeline = pipeline[model_name][field_name].pop(
                    "profile_pipeline"
                )
                triple_type = pipeline[model_name][field_name].pop("triple_type")
                # get relationships
                relationship_extractor = create_dynamic_relationship_extraction(
                    specification=specification,
                    triple_types=triple_type,
                    elasticsearch=resources["elasticsearch"],
                    index_name=resources["index_name"],
                )
                relationships = relationship_extractor.extract_parallel()
                # clean the graph to get the deduplicated relationships
                deduplicated_graph = create_deduplicated_graph(
                    relationships=relationships
                )
                profiled_relationships = parallel_process_relationships(
                    deduplicated_graph=deduplicated_graph,
                    profile_pipeline=profile_pipeline,
                    max_workers=max_workers,
                    quality_threshold=quality_threshold,
                    generate_profiles=generate_relationship_profiles,
                )
                values[model_name][field_name] = profiled_relationships
    return values


def collect_single_value(
    specification: str,
    pipeline: dict,
    model_name: str,
    field_name: str,
    entry: Union[tuple[Type[WebSearchRAG], str], dict],
    resources: dict,
    max_workers: int,
    quality_threshold: int,
    generate_relationship_profiles: bool,
):
    if isinstance(entry, tuple):  # handle single value maps
        value = run_specified_rag(
            specification=specification,
            name=field_name,
            description=entry[1],  # value_description
            rag=entry[0],  # WebSearchValueRAG subclass
        )
        return {field_name: value}
    if isinstance(entry, dict):  # handle nested value maps with relationship extraction
        profile_pipeline = pipeline[model_name][field_name].pop("profile_pipeline")
        triple_type = pipeline[model_name][field_name].pop("triple_type")
        # get relationships
        relationship_extractor = create_dynamic_relationship_extraction(
            specification=specification,
            triple_types=triple_type,
            elasticsearch=resources["elasticsearch"],
            index_name=resources["index_name"],
        )
        relationships = relationship_extractor.extract_parallel()
        # clean the graph to get the deduplicated relationships
        deduplicated_graph = create_deduplicated_graph(relationships=relationships)
        profiled_relationships = parallel_process_relationships(
            deduplicated_graph=deduplicated_graph,
            profile_pipeline=profile_pipeline,
            max_workers=max_workers,
            quality_threshold=quality_threshold,
            generate_profiles=generate_relationship_profiles,
        )
        return {field_name: profiled_relationships}


def run_value_rag_pipeline_parallel(
    specification: str,
    pipeline: dict[str, tuple[Type[WebSearchRAG], str]],
    max_workers: int = 4,
    quality_threshold: int = 3,
    generate_relationship_profiles: bool = False,
) -> dict[str, list[dict[str, Type[CitedValueWithCredibility]]]]:
    """
    Run a pipeline of WebSearchValueRAG classes based on the provided value map.
    """
    resources = pipeline.pop("_resources")
    values = {}
    futures = []
    for model_name in pipeline:
        values[model_name] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for field_name, entry in pipeline[model_name].items():
                future = executor.submit(
                    collect_single_value,
                    specification,
                    pipeline,
                    model_name,
                    field_name,
                    entry,
                    resources,
                    max_workers,
                    quality_threshold,
                    generate_relationship_profiles,
                )
                futures.append(future)
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                values[model_name].update(result)
    return values
