"""
Pipelines for creating vector stores from s3 data store
"""
from abc import ABC, abstractmethod
from conductor.database.pinecone_ import (
    ConductorJobS3Pipeline,
    ConductorBulkS3Pipeline,
    ApolloS3Pipeline,
    DiscordS3Pipeline,
    ApolloBulkS3Pipeline,
    ApifyBulkS3Pipeline,
)
from pinecone import Pinecone, ServerlessSpec
from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
import os


class PineconeCreateDestroyUpdatePipeline(ABC):
    """
    Handle the creation, building, and destruction of Pinecone vector stores
    """

    def __init__(self) -> None:
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    def _create_vectorstore(
        self, index_name: str, dimension: int, metric: str, cloud: str, region: str
    ) -> None:
        return self.pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    def _update_vectorstore(
        self,
        index_name: str,
        source_bucket_name: str,
        job_id: str,
        conductor_job_s3_pipeline: ConductorJobS3Pipeline,
        embedding_function: Embeddings,
    ):
        conductor_job_s3_pipeline = conductor_job_s3_pipeline(
            source_bucket_name=source_bucket_name,
            destination_index=index_name,
            embedding_function=embedding_function,
            job_id=job_id,
        )
        return conductor_job_s3_pipeline.add_documents_from_job()

    def _destroy_vector_store(self, index_name: str):
        return self.pinecone.delete_index(name=index_name)

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def destroy(self):
        pass

    @abstractmethod
    def update(self, job_id: str):
        pass


class BulkPineconeCreateDestroyPipeline(ABC):
    """
    Handle the bulk creation, building, and destruction of Pinecone vector stores
    """

    def __init__(self) -> None:
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    def _create_vectorstore(
        self, index_name: str, dimension: int, metric: str, cloud: str, region: str
    ) -> None:
        return self.pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    def _build_vectorstore(
        self,
        index_name: str,
        source_bucket_name: str,
        conductor_s3_pipeline: ConductorBulkS3Pipeline,
        embedding_function: Embeddings,
    ):
        conductor_s3_pipeline = conductor_s3_pipeline(
            source_bucket_name=source_bucket_name,
            destination_index=index_name,
            embedding_function=embedding_function,
        )
        return conductor_s3_pipeline.add_documents_from_bucket()

    def _destroy_vector_store(self, index_name: str):
        return self.pinecone.delete_index(name=index_name)

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def destroy(self):
        pass

    @abstractmethod
    def build(self):
        pass


class ApolloPineconeCreateDestroyUpdatePipeline(PineconeCreateDestroyUpdatePipeline):
    """
    Pipeline for creating and destroying Apollo Pinecone vector stores
    These pipelines read from templated values in conductor environment variables
    """

    def create(self):
        return self._create_vectorstore(
            index_name=os.getenv("PINECONE_APOLLO_INDEX"),
            dimension=int(os.getenv("PINECONE_VECTOR_DIMENSIONS")),
            metric=os.getenv("PINECONE_METRIC"),
            cloud=os.getenv("PINECONE_CLOUD"),
            region=os.getenv("PINECONE_REGION"),
        )

    def destroy(self):
        return self._destroy_vector_store(index_name=os.getenv("PINECONE_APOLLO_INDEX"))

    def update(
        self, job_id: str, index_name: str = None, source_bucket_name: str = None
    ):
        return self._update_vectorstore(
            index_name=index_name if index_name else os.getenv("PINECONE_APOLLO_INDEX"),
            source_bucket_name=source_bucket_name
            if source_bucket_name
            else os.getenv("CONDUCTOR_S3_BUCKET"),
            job_id=job_id,
            conductor_job_s3_pipeline=ApolloS3Pipeline,
            embedding_function=BedrockEmbeddings(
                region_name=os.getenv("BEDROCK_REGION"),
            ),
        )


class DiscordPineconeCreateDestroyUpdatePipeline(PineconeCreateDestroyUpdatePipeline):
    """
    Pipeline for creating and destroying Discord Pinecone vector stores
    These pipelines read from templated values in conductor environment variables
    """

    def create(self, index_name: str = None):
        return self._create_vectorstore(
            index_name=index_name
            if index_name
            else os.getenv("PINECONE_DISCORD_INDEX"),
            dimension=int(os.getenv("PINECONE_VECTOR_DIMENSIONS")),
            metric=os.getenv("PINECONE_METRIC"),
            cloud=os.getenv("PINECONE_CLOUD"),
            region=os.getenv("PINECONE_REGION"),
        )

    def destroy(self):
        return self._destroy_vector_store(
            index_name=os.getenv("PINECONE_DISCORD_INDEX")
        )

    def update(
        self, job_id: str, index_name: str = None, source_bucket_name: str = None
    ):
        return self._update_vectorstore(
            index_name=index_name,
            source_bucket_name=source_bucket_name,
            job_id=job_id,
            conductor_job_s3_pipeline=DiscordS3Pipeline,
            embedding_function=BedrockEmbeddings(
                region_name=os.getenv("BEDROCK_REGION"),
            ),
        )


class ApolloBulkPineconeCreateDestroyPipeline(BulkPineconeCreateDestroyPipeline):
    """
    Pipeline for creating and destroying Apollo Bulk Pinecone vector stores
    These pipelines read from templated values in conductor environment variables
    """

    def create(self):
        return self._create_vectorstore(
            index_name=os.getenv("PINECONE_APOLLO_INDEX"),
            dimension=int(os.getenv("PINECONE_VECTOR_DIMENSIONS")),
            metric=os.getenv("PINECONE_METRIC"),
            cloud=os.getenv("PINECONE_CLOUD"),
            region=os.getenv("PINECONE_REGION"),
        )

    def destroy(self):
        return self._destroy_vector_store(index_name=os.getenv("PINECONE_APOLLO_INDEX"))

    def build(self):
        return self._build_vectorstore(
            index_name=os.getenv("PINECONE_APOLLO_INDEX"),
            source_bucket_name=os.getenv("CONDUCTOR_S3_BUCKET"),
            conductor_s3_pipeline=ApolloBulkS3Pipeline,
            embedding_function=BedrockEmbeddings(
                region_name=os.getenv("BEDROCK_REGION"),
            ),
        )


class ApifyBulkPineconeCreateDestroyPipeline(BulkPineconeCreateDestroyPipeline):
    """Bulk pipeline for creating and destroying Apify Bulk Pinecone vector stores

    Args:
        BulkPineconeCreateDestroyPipeline (_type_): _description_
    """

    def create(self):
        return self._create_vectorstore(
            index_name=os.getenv("PINECONE_APIFY_INDEX"),
            dimension=int(os.getenv("PINECONE_VECTOR_DIMENSIONS")),
            metric=os.getenv("PINECONE_METRIC"),
            cloud=os.getenv("PINECONE_CLOUD"),
            region=os.getenv("PINECONE_REGION"),
        )

    def destroy(self):
        return self._destroy_vector_store(index_name=os.getenv("PINECONE_APIFY_INDEX"))

    def build(self, index_name: str = None, source_bucket_name: str = None):
        return self._build_vectorstore(
            index_name=index_name if index_name else os.getenv("PINECONE_APIFY_INDEX"),
            source_bucket_name=source_bucket_name
            if source_bucket_name
            else os.getenv("APIFY_S3_BUCKET"),
            conductor_s3_pipeline=ApifyBulkS3Pipeline,
            embedding_function=BedrockEmbeddings(
                region_name=os.getenv("BEDROCK_REGION"),
            ),
        )
