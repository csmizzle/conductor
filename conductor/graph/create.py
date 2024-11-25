"""
Implementation of more comprehensive deduplication of relationships when looking across multiple triple types
"""
from conductor.graph import models
from conductor.flow.rag import DocumentWithCredibility
import hashlib
from typing import List


def create_deduplicated_graph(
    relationships: List[models.CitedRelationshipWithCredibility],
) -> models.AggregatedCitedGraph:
    """
    Create deduplicated graph
    """
    dedupe_dict: dict[str, list[int]] = {}  # hash to indexes
    # deduplicate relationships through normalization and dedupe with polars
    for idx, relationship in enumerate(relationships):
        # normalize source, relationship, and target
        source_type = relationship.source.entity_type
        source = relationship.source.name.lower()
        relationship_type = relationship.relationship_type
        target_type = relationship.target.entity_type
        target = relationship.target.name.lower()
        # hash normalized relationship
        hash_obj = hashlib.md5()
        hash_obj.update(
            f"{source_type}{source}{relationship_type}{target_type}{target}".encode()
        )
        hash_str = hash_obj.hexdigest()
        if hash_str not in dedupe_dict:
            dedupe_dict[hash_str] = [idx]
        else:
            dedupe_dict[hash_str].append(idx)
    # iterate over hash values and deduplicate with aggregated values
    cleaned_entities_map = {}
    cleaned_cited_entities: list[models.AggregatedCitedEntity] = []
    cleaned_relationships: list[models.AggregatedCitedRelationship] = []
    aggregated_documents: dict[str, list[DocumentWithCredibility]] = {}
    for entry in dedupe_dict:
        # handle case where there is only one relationship
        if len(dedupe_dict[entry]) == 1:
            # add source entity to cleaned entities map by using index of the relationship
            if (
                relationships[dedupe_dict[entry][0]].source.name
                not in cleaned_entities_map
            ):
                cleaned_entities_map[
                    relationships[dedupe_dict[entry][0]].source.name
                ] = relationships[dedupe_dict[entry][0]].source
                cleaned_cited_entities.append(
                    models.AggregatedCitedEntity(
                        entity=relationships[dedupe_dict[entry][0]].source,
                        documents=[
                            DocumentWithCredibility(
                                content=relationships[
                                    dedupe_dict[entry][0]
                                ].document_content,
                                source=relationships[
                                    dedupe_dict[entry][0]
                                ].document_source,
                                source_credibility=relationships[
                                    dedupe_dict[entry][0]
                                ].document_source_credibility,
                                source_credibility_reasoning=relationships[
                                    dedupe_dict[entry][0]
                                ].document_source_credibility_reasoning,
                            )
                        ],
                    )
                )
            if (
                relationships[dedupe_dict[entry][0]].target.name
                not in cleaned_entities_map
            ):
                cleaned_entities_map[
                    relationships[dedupe_dict[entry][0]].target.name
                ] = relationships[dedupe_dict[entry][0]].target
                cleaned_cited_entities.append(
                    models.AggregatedCitedEntity(
                        entity=relationships[dedupe_dict[entry][0]].target,
                        documents=[
                            DocumentWithCredibility(
                                content=relationships[
                                    dedupe_dict[entry][0]
                                ].document_content,
                                source=relationships[
                                    dedupe_dict[entry][0]
                                ].document_source,
                                source_credibility=relationships[
                                    dedupe_dict[entry][0]
                                ].document_source_credibility,
                                source_credibility_reasoning=relationships[
                                    dedupe_dict[entry][0]
                                ].document_source_credibility_reasoning,
                            )
                        ],
                    )
                )
            # add relationship to cleaned relationships
            cleaned_relationships.append(
                models.AggregatedCitedRelationship(
                    source=relationships[dedupe_dict[entry][0]].source,
                    target=relationships[dedupe_dict[entry][0]].target,
                    relationship_type=relationships[
                        dedupe_dict[entry][0]
                    ].relationship_type,
                    relationship_reasoning=relationships[
                        dedupe_dict[entry][0]
                    ].relationship_reasoning,
                    relationship_faithfulness=relationships[
                        dedupe_dict[entry][0]
                    ].relationship_faithfulness,
                    relationship_factual_correctness=relationships[
                        dedupe_dict[entry][0]
                    ].relationship_factual_correctness,
                    relationship_confidence=relationships[
                        dedupe_dict[entry][0]
                    ].relationship_confidence,
                    relationships_query=relationships[
                        dedupe_dict[entry][0]
                    ].relationships_query,
                    documents=[
                        DocumentWithCredibility(
                            content=relationships[
                                dedupe_dict[entry][0]
                            ].document_content,
                            source=relationships[dedupe_dict[entry][0]].document_source,
                            source_credibility=relationships[
                                dedupe_dict[entry][0]
                            ].document_source_credibility,
                            source_credibility_reasoning=relationships[
                                dedupe_dict[entry][0]
                            ].document_source_credibility_reasoning,
                        )
                    ],
                )
            )
        # aggregate documents
        else:
            # first pass to aggregate documents
            for idx in dedupe_dict[entry]:
                if entry not in aggregated_documents:
                    aggregated_documents[entry] = [
                        DocumentWithCredibility(
                            content=relationships[idx].document_content,
                            source=relationships[idx].document_source,
                            source_credibility=relationships[
                                idx
                            ].document_source_credibility,
                            source_credibility_reasoning=relationships[
                                idx
                            ].document_source_credibility_reasoning,
                        )
                    ]
                else:
                    aggregated_documents[entry].append(
                        DocumentWithCredibility(
                            content=relationships[idx].document_content,
                            source=relationships[idx].document_source,
                            source_credibility=relationships[
                                idx
                            ].document_source_credibility,
                            source_credibility_reasoning=relationships[
                                idx
                            ].document_source_credibility_reasoning,
                        )
                    )
            # add source entity to cleaned entities map by using index of the relationship
            for idx in dedupe_dict[entry]:
                if relationships[idx].source.name not in cleaned_entities_map:
                    cleaned_entities_map[
                        relationships[idx].source.name
                    ] = relationships[idx].source
                    cleaned_cited_entities.append(
                        models.AggregatedCitedEntity(
                            entity=relationships[idx].source,
                            documents=aggregated_documents[entry],
                        )
                    )
                if relationships[idx].target.name not in cleaned_entities_map:
                    cleaned_entities_map[
                        relationships[idx].target.name
                    ] = relationships[idx].target
                    cleaned_cited_entities.append(
                        models.AggregatedCitedEntity(
                            entity=relationships[idx].target,
                            documents=aggregated_documents[entry],
                        )
                    )
    # populate for all relationships and aggregated documents
    for entry in aggregated_documents:
        # on duplicate entries, take the first source and target and grab aggregated documents
        cleaned_relationships.append(
            models.AggregatedCitedRelationship(
                source=relationships[dedupe_dict[entry][0]].source,
                target=relationships[dedupe_dict[entry][0]].target,
                relationship_type=relationships[
                    dedupe_dict[entry][0]
                ].relationship_type,
                relationship_reasoning=relationships[
                    dedupe_dict[entry][0]
                ].relationship_reasoning,
                relationship_faithfulness=relationships[
                    dedupe_dict[entry][0]
                ].relationship_faithfulness,
                relationship_factual_correctness=relationships[
                    dedupe_dict[entry][0]
                ].relationship_factual_correctness,
                relationship_confidence=relationships[
                    dedupe_dict[entry][0]
                ].relationship_confidence,
                relationships_query=relationships[
                    dedupe_dict[entry][0]
                ].relationships_query,
                documents=aggregated_documents[entry],
            )
        )
    return models.AggregatedCitedGraph(
        entities=cleaned_cited_entities,
        relationships=cleaned_relationships,
    )
