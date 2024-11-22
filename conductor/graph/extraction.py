"""
Build out relationship and graph extraction utilities here
- Specification for entity extraction
- Start with list of triple types
- Iterate over relationships
    - Build query for each relationship
    - Return list of documents
    - Iterate over documents
        - Extract entities
        - Extract relationships
- Return graph
"""
from typing import List, Tuple, Optional, Union
from conductor.graph.models import TripleType, Entity, Relationship, Graph
from conductor.graph import signatures
from conductor.flow.credibility import SourceCredibility
from conductor.flow.rag import CitedAnswerWithCredibility
import concurrent.futures
import dspy
from pydantic import InstanceOf, BaseModel, Field
from loguru import logger
import polars as pl


class CitedRelationshipWithCredibility(BaseModel):
    """
    Cited relationship model
    """

    source: Entity = Field(description="Source entity")
    target: Entity = Field(description="Target entity")
    relationship_type: str = Field(description="Relationship type")
    # individual relationship metadata
    relationship_reasoning: str = Field(
        description="The reasoning behind the relationship"
    )
    relationship_faithfulness: int = Field(
        ge=1, le=5, description="The faithfulness of the relationship"
    )
    relationship_factual_correctness: int = Field(
        ge=1, le=5, description="The factual correctness of the relationship"
    )
    relationship_confidence: int = Field(
        ge=1, le=5, description="The confidence of the relationship"
    )
    # relationship extraction metadata
    document: str = Field(description="The document used to generate the relationship")
    relationships_query: str = Field(
        description="The query used to generate the relationship"
    )
    relationships_reasoning: str = Field(
        description="The reasoning behind the relationship"
    )
    # document collection metadata
    question: str = Field(description="The question")
    answer: str = Field(description="The answer for the question")
    documents: list[str] = Field(
        description="The documents used to generate the answer"
    )
    answer_reasoning: Union[str, None] = Field(
        description="The reasoning behind the answer"
    )
    citations: list[str] = Field(description="The URLs used in the answer")
    faithfulness: int = Field(ge=1, le=5, description="The faithfulness of the answer")
    factual_correctness: int = Field(
        ge=1, le=5, description="The factual correctness of the answer"
    )
    confidence: int = Field(ge=1, le=5, description="The confidence of the answer")
    source_credibility: list[SourceCredibility] = Field(
        description="The credibility of the sources"
    )
    source_credibility_reasoning: Optional[list[str]] = Field(
        description="The reasoning behind the source credibility"
    )

    class Config:
        use_enum_values = True


class RelationshipRAGExtractor:
    """
    Graph extraction utilities
    """

    def __init__(
        self,
        specification: str,
        triple_types: List[TripleType],
        rag: InstanceOf[dspy.Module],
    ) -> None:
        self.specification = specification
        self.triple_types = triple_types
        self.rag = rag
        self.create_relationship_query = dspy.ChainOfThought(
            signatures.RelationshipQuery
        )
        self.extract_relationships = dspy.ChainOfThought(
            signatures.ExtractedRelationships
        )
        self.create_relationship_reasoning = dspy.ChainOfThought(
            signatures.RelationshipReasoning
        )

    def extract(self) -> list[CitedRelationshipWithCredibility]:
        """
        Extract graph
        """
        relationships = []
        for triple_type in self.triple_types:
            query = self.create_relationship_query(
                specification=self.specification,
                triple_type=triple_type,
            )
            logger.info(f"Query: {query.query}")
            answer: CitedAnswerWithCredibility = self.rag(question=query.query)
            logger.info(f"Retrieved {len(answer.documents)} documents")
            for document in answer.documents:
                extracted_relationships = self.extract_relationships(
                    query=answer.question, document=document, triple_type=triple_type
                )
                logger.info(
                    f"Extracted {len(extracted_relationships.relationships)} relationships"
                )
                # map to cited relationship schema
                for relationship in extracted_relationships.relationships:
                    # create relationship reasoning from query, relationship, and document
                    logger.info("Creating relationship reasoning ...")
                    relationship_reasoning = self.create_relationship_reasoning(
                        query=answer.question,
                        relationship=relationship,
                        document=document,
                    )
                    relationships.append(
                        CitedRelationshipWithCredibility(
                            source=relationship.source,
                            target=relationship.target,
                            relationship_reasoning=relationship_reasoning.relationship_reasoning,
                            relationship_type=relationship.relationship_type,
                            relationship_faithfulness=relationship.faithfulness,
                            relationship_factual_correctness=relationship.factual_correctness,
                            relationship_confidence=relationship.confidence,
                            document=document,
                            relationships_query=answer.question,
                            relationships_reasoning=extracted_relationships.reasoning,
                            question=answer.question,
                            answer=answer.answer,
                            documents=answer.documents,
                            answer_reasoning=answer.answer_reasoning,
                            citations=answer.citations,
                            faithfulness=answer.faithfulness,
                            factual_correctness=answer.factual_correctness,
                            confidence=answer.confidence,
                            source_credibility=answer.source_credibility,
                            source_credibility_reasoning=answer.source_credibility_reasoning,
                        )
                    )
        return relationships

    def _create_relationship_query(
        self, triple_type: TripleType, specification: str
    ) -> str:
        """
        Light wrapper around create_relationship_query to add logging
        """
        query = self.create_relationship_query(
            specification=specification, triple_type=triple_type
        )
        logger.info(f"Query: {query.query}")
        return query.query

    def _execute_query(self, query: str) -> list[str]:
        """
        Light wrapper around retriever to add logging
        """
        documents = self.retriever(query=query)
        logger.info(f"Retrieved {len(documents.documents)} documents")
        return documents.documents

    def _execute_queries_parallel(
        self,
    ) -> Tuple[dict[str, TripleType], dict[str, list[str]]]:
        """
        Execute query generation and execution in parallel
        Returns queries and documents for extraction algorithm
        """
        queries: dict[str, TripleType] = {}
        # generate all queries
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            # map result to triple type
            for idx, triple_type in enumerate(self.triple_types):
                futures[idx] = executor.submit(
                    self._create_relationship_query, triple_type, self.specification
                )
            # map query to triple type
            for triple_type_idx, future in futures.items():
                queries[future.result()] = self.triple_types[triple_type_idx]
        # execute all queries
        documents: dict[str, list[str]] = {}  # map query to documents
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for query in queries:
                futures[query] = executor.submit(self._execute_query, query)
            for query, future in futures.items():
                documents[query] = future.result()
        return queries, documents

    def _extract_relationships(
        self, query: str, document: str, triple_type: TripleType
    ) -> list[Relationship]:
        """
        Light wrapper around extract_relationships to add logging
        """
        extracted_relationships = self.extract_relationships(
            query=query, document=document, triple_type=triple_type
        )
        logger.info(
            f"Extracted {len(extracted_relationships.relationships)} relationships"
        )
        return extracted_relationships.relationships

    def _extract_relationships_parallel(
        self, queries: dict[str, TripleType], documents: dict[str, list[str]]
    ) -> list[Relationship]:
        """
        Extract relationships in parallel
        We use the query maps to execute the correct extraction for each set of documents
        """
        relationships = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for query in documents:
                for document in documents[query]:
                    futures.append(
                        executor.submit(
                            self._extract_relationships,
                            query=query,
                            document=document,
                            triple_type=queries[query],
                        )
                    )
            for future in concurrent.futures.as_completed(futures):
                relationships.extend(future.result())
        return relationships

    def extract_parallel(self) -> list[Relationship]:
        # first execute all queries in parallel
        queries, documents = self._execute_queries_parallel()
        # then extract relationships in parallel
        extracted_relationships = self._extract_relationships_parallel(
            queries=queries, documents=documents
        )
        logger.info(f"Extracted {len(extracted_relationships)} relationships")
        return extracted_relationships


def create_graph(
    specification: str,
    triple_types: List[TripleType],
    rag: InstanceOf[dspy.Module],
) -> dict:
    """
    Create graph
    """
    extractor = RelationshipRAGExtractor(
        specification=specification, triple_types=triple_types, rag=rag
    )
    relationships = extractor.extract_parallel()
    # deduplicate relationships through normalization and dedup with polars
    normalized_relationships = []
    for relationship in relationships:
        # normalize source, relationship, and target
        source_type = relationship.source.entity_type
        source = relationship.source.name.lower()
        relationship_type = relationship.relationship_type
        target_type = relationship.target.entity_type
        target = relationship.target.name.lower()
        normalized_relationships.append(
            {
                "source_type": source_type,
                "source": source,
                "relationship_type": relationship_type,
                "target_type": target_type,
                "target": target,
            }
        )
    df = pl.DataFrame(normalized_relationships)
    logger.info(f"Deduplicating {len(df)} relationships")
    df = df.unique()
    deduped_relationships = df.to_dicts()
    logger.info(f"Deduplicated to {len(deduped_relationships)} relationships")
    # map relationships to graph
    cleaned_relationships = []
    cleaned_entities_map = {}
    for relationship in deduped_relationships:
        if relationship["source"] not in cleaned_entities_map:
            cleaned_entities_map[relationship["source"]] = Entity(
                entity_type=relationship["source_type"], name=relationship["source"]
            )
        if relationship["target"] not in cleaned_entities_map:
            cleaned_entities_map[relationship["target"]] = Entity(
                entity_type=relationship["target_type"], name=relationship["target"]
            )
        cleaned_relationships.append(
            Relationship(
                source=cleaned_entities_map[relationship["source"]],
                target=cleaned_entities_map[relationship["target"]],
                relationship_type=relationship["relationship_type"],
            )
        )
    logger.info(
        f"Created graph with {len(cleaned_entities_map)} entities and {len(cleaned_relationships)} relationships"
    )
    return Graph(
        entities=list(cleaned_entities_map.values()),
        relationships=cleaned_relationships,
    )
