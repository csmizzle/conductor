"""
Generate RAG based profile building
"""
from pydantic import InstanceOf
from conductor.profiles.utils import specify_model
from conductor.flow.rag import CitedValueWithCredibility
import concurrent.futures
import dspy
from loguru import logger


def generate_profile(
    model_schema: dict, specification: str, rag: InstanceOf[dspy.Module]
) -> dict[str, CitedValueWithCredibility]:
    """
    Generate a profile based on a model and a specification
    """
    specified_fields = specify_model(
        model_schema=model_schema, specification=specification
    )
    profile = {}
    for field in specified_fields:
        logger.info(f"Gathering information for field question: {field}")
        profile[field] = rag(question=specified_fields[field])
    return profile


def generate_profile_parallel(
    model_schema: dict, specification: str, rag: InstanceOf[dspy.Module]
) -> dict[str, CitedValueWithCredibility]:
    """
    Generate a profile based on a model and a specification
    """
    specified_fields = specify_model(
        model_schema=model_schema, specification=specification
    )
    profile = {}
    # Use concurrent futures to parallelize the profile generation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(rag, question=specified_fields[field]): field
            for field in specified_fields
        }
        for future in concurrent.futures.as_completed(futures):
            field = futures[future]
            profile[field] = future.result()
    return profile
