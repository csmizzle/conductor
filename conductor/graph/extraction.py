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
from typing import List, Tuple
from conductor.graph.models import TripleType, Entity, Relationship, Graph
from conductor.flow.retriever import ElasticRMClient
from conductor.graph import signatures
import concurrent.futures
import dspy
from loguru import logger
import polars as pl


class RelationshipRAGExtractor:
    """
    Graph extraction utilities
    """

    def __init__(
        self,
        specification: str,
        triple_types: List[TripleType],
        retriever: ElasticRMClient,
    ) -> None:
        self.specification = specification
        self.triple_types = triple_types
        self.retriever = retriever
        self.create_relationship_query = dspy.ChainOfThought(
            signatures.RelationshipQuery
        )
        self.extract_relationships = dspy.ChainOfThought(
            signatures.ExtractedRelationships
        )

    def extract(self) -> list[Relationship]:
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
            documents = self.retriever(query=query.query)
            logger.info(f"Retrieved {len(documents.documents)} documents")
            for document in documents.documents:
                extracted_relationships = self.extract_relationships(
                    query=query.query, document=document, triple_type=triple_type
                )
                logger.info(
                    f"Extracted {len(extracted_relationships.relationships)} relationships"
                )
                relationships.extend(extracted_relationships.relationships)
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
    retriever: ElasticRMClient,
) -> dict:
    """
    Create graph
    """
    extractor = RelationshipRAGExtractor(
        specification=specification, triple_types=triple_types, retriever=retriever
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
