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
.. with some optimizations
"""
from typing import List, Tuple
from conductor.graph import models
from conductor.graph import signatures
from conductor.flow.rag import DocumentWithCredibility
import concurrent.futures
import dspy
from pydantic import InstanceOf
from loguru import logger


class RelationshipRAGExtractor:
    """
    Graph extraction utilities
    """

    def __init__(
        self,
        specification: str,
        triple_types: List[models.TripleType],
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

    def extract(self) -> list[models.CitedRelationshipWithCredibility]:
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
            documents: list[DocumentWithCredibility] = self.rag(question=query.query)
            logger.info(f"Retrieved {len(documents)} documents")
            for document in documents:
                extracted_relationships = self.extract_relationships(
                    query=query.query, document=document, triple_type=triple_type
                )
                logger.info(
                    f"Extracted {len(extracted_relationships.relationships)} relationships"
                )
                # map to cited relationship schema
                for relationship in extracted_relationships.relationships:
                    # create relationship reasoning from query, relationship, and document
                    logger.info("Creating relationship reasoning ...")
                    relationship_reasoning = self.create_relationship_reasoning(
                        query=query.query,
                        relationship=relationship,
                        document=document,
                    )
                    relationships.append(
                        models.CitedRelationshipWithCredibility(
                            source=relationship.source,
                            target=relationship.target,
                            relationship_type=relationship.relationship_type,
                            relationship_reasoning=relationship_reasoning.relationship_reasoning,
                            relationship_faithfulness=relationship.faithfulness,
                            relationship_factual_correctness=relationship.factual_correctness,
                            relationship_confidence=relationship.confidence,
                            relationships_query=query.query,
                            document_content=document.content,
                            document_source=document.source,
                            document_source_credibility=document.source_credibility,
                            document_source_credibility_reasoning=document.source_credibility_reasoning,
                        )
                    )
        return relationships

    def _create_relationship_query(
        self, triple_type: models.TripleType, specification: str
    ) -> str:
        """
        Light wrapper around create_relationship_query to add logging
        """
        query = self.create_relationship_query(
            specification=specification, triple_type=triple_type
        )
        logger.info(f"Query: {query.query}")
        return query.query

    def _execute_query(self, query: str) -> list[DocumentWithCredibility]:
        """
        Light wrapper around retriever to add logging
        """
        documents = self.rag(question=query)
        logger.info(f"Retrieved {len(documents)} documents")
        return documents

    def _execute_queries_parallel(
        self,
    ) -> Tuple[dict[str, models.TripleType], dict[str, list[DocumentWithCredibility]]]:
        """
        Execute query generation and execution in parallel
        Returns queries and documents for extraction algorithm
        """
        queries: dict[str, models.TripleType] = {}
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
        documents: dict[
            str, list[DocumentWithCredibility]
        ] = {}  # map query to documents
        # same idea here, map result to query for downstream operations
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for query in queries:
                futures[query] = executor.submit(self._execute_query, query)
            for query, future in futures.items():
                documents[query] = future.result()
        return queries, documents

    def _extract_relationships(
        self,
        query: str,
        document: DocumentWithCredibility,
        triple_type: models.TripleType,
    ) -> Tuple[str, list[models.Relationship]]:
        """
        Light wrapper around extract_relationships to add logging also return document for downstream reasoning
        """
        extracted_relationships = self.extract_relationships(
            query=query, document=document, triple_type=triple_type
        )
        logger.info(
            f"Extracted {len(extracted_relationships.relationships)} relationships"
        )
        return document, extracted_relationships.relationships

    def _extract_relationships_parallel(
        self,
        queries: dict[str, models.TripleType],
        documents: dict[str, list[DocumentWithCredibility]],
    ) -> dict[str, list[Tuple[DocumentWithCredibility, list[models.Relationship]]]]:
        """
        Extract relationships in parallel
        We use the query maps to execute the correct extraction for each set of documents
        """
        relationships: dict[
            str, list[Tuple[DocumentWithCredibility, list[models.Relationship]]]
        ] = {}  # map query to document and relationships
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures: dict[str, list] = {}
            for query in documents:
                for document in documents[query]:
                    if query not in futures:
                        futures[query] = [
                            executor.submit(
                                self._extract_relationships,
                                query=query,
                                document=document,
                                triple_type=queries[query],
                            )
                        ]
                    else:
                        futures[query].append(
                            executor.submit(
                                self._extract_relationships,
                                query=query,
                                document=document,
                                triple_type=queries[query],
                            )
                        )
            for query in futures:
                # get document and relationships from future
                for future in futures[query]:
                    document, extracted_relationships = future.result()
                    if query not in relationships:
                        relationships[query] = [(document, extracted_relationships)]
                    else:
                        relationships[query].append((document, extracted_relationships))
        return relationships

    def _create_relationship_reasoning(
        self,
        relationship: models.Relationship,
        query: str,
        document: DocumentWithCredibility,
    ) -> Tuple[
        str, DocumentWithCredibility, models.Relationship, str
    ]:  # query, document, relationship, reasoning
        """
        Light wrapper around create_relationship_reasoning to add logging
        """
        reasoning = self.create_relationship_reasoning(
            query=query, relationship=relationship, document=document
        )
        logger.info(f"Reasoning: {reasoning.relationship_reasoning}")
        return query, document, relationship, reasoning.relationship_reasoning

    def _create_relationships_parallel(
        self,
        relationships: dict[
            str, list[Tuple[DocumentWithCredibility, list[models.Relationship]]]
        ],
    ) -> list[models.CitedRelationshipWithCredibility]:
        """
        Create relationship reasoning in parallel and then map to cited relationship schema from answers
        """
        extracted_relationships = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures: dict[str, list] = {}
            for query in relationships:
                for entry in relationships[query]:
                    for relationship in entry[1]:
                        if query not in futures:
                            futures[query] = [
                                executor.submit(
                                    self._create_relationship_reasoning,
                                    relationship=relationship,
                                    query=query,
                                    document=entry[0],
                                )
                            ]
                        else:
                            futures[query].append(
                                executor.submit(
                                    self._create_relationship_reasoning,
                                    relationship=relationship,
                                    query=query,
                                    document=entry[0],
                                )
                            )
            for query in futures:
                for future in futures[query]:
                    query, document, relationship, reasoning = future.result()
                    extracted_relationships.append(
                        models.CitedRelationshipWithCredibility(
                            source=relationship.source,
                            target=relationship.target,
                            relationship_type=relationship.relationship_type,
                            relationship_reasoning=reasoning,
                            relationship_faithfulness=relationship.faithfulness,
                            relationship_factual_correctness=relationship.factual_correctness,
                            relationship_confidence=relationship.confidence,
                            relationships_query=query,
                            document_content=document.content,
                            document_source=document.source,
                            document_source_credibility=document.source_credibility,
                            document_source_credibility_reasoning=document.source_credibility_reasoning,
                        )
                    )
        return extracted_relationships

    def extract_parallel(self) -> list[models.CitedRelationshipWithCredibility]:
        # first execute all queries in parallel
        queries, documents = self._execute_queries_parallel()
        # then extract relationships in parallel
        extracted_relationships = self._extract_relationships_parallel(
            queries=queries, documents=documents
        )
        # reason and map to cited relationship schema
        extracted_relationships = self._create_relationships_parallel(
            relationships=extracted_relationships
        )
        logger.info(f"Extracted {len(extracted_relationships)} relationships")
        return extracted_relationships
