"""
Pinecone vector databases
- To start, let's vectorize the discord and apollo data
- Make sure to also have the data and metadata
"""
from abc import ABC, abstractmethod
from langchain_core.embeddings.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
import boto3
import json
import logging

logger = logging.getLogger(__name__)


class ConductorJobS3Pipeline(ABC):
    """
    Take a given job from s3 and add it to a Pinecone index
    """

    def __init__(
        self,
        source_bucket_name: str,
        job_id: str,
        destination_index: str,
        embedding_function: Embeddings = None,
    ) -> None:
        self.source_bucket_name = source_bucket_name
        self.job_id = job_id
        self.destination_index = destination_index
        self.embedding_function = embedding_function

    def get_data_from_job(
        self, job_id_suffix: str = None, file_ends_with: str = None
    ) -> list[dict]:
        """
        Get all files from an Discord job
        """
        if job_id_suffix:
            prefix = self.job_id + job_id_suffix
        else:
            prefix = self.job_id
        if file_ends_with:
            object_ends_with = file_ends_with
        else:
            object_ends_with = ".json"
        s3 = boto3.client("s3")
        bucket = boto3.resource("s3").Bucket(self.source_bucket_name)
        data = bucket.objects.filter(Prefix=prefix)
        for object in data:
            if object.key.endswith(object_ends_with):
                response = s3.get_object(Bucket=self.source_bucket_name, Key=object.key)
                return json.loads(response["Body"].read().decode("utf-8"))

    @abstractmethod
    def prepare_documents_for_pinecone(self) -> list[str]:
        pass

    def add_documents_from_job(self) -> None:
        """
        Vectorize text using the embedding function
        """
        documents = self.prepare_documents_for_pinecone()
        # create langchain documents
        # vectorize the documents
        pinecone_index = PineconeVectorStore(
            index_name=self.destination_index, embedding=self.embedding_function
        )
        pinecone_index.add_documents(documents)
        print(f"Indexed {len(documents)} documents to {self.destination_index}")


class ConductorBulkS3Pipeline(ABC):
    """
    Take all jobs from s3 and add them to a Pinecone index
    """

    def __init__(
        self,
        source_bucket_name: str,
        destination_index: str,
        embedding_function: Embeddings = None,
    ) -> None:
        self.source_bucket_name = source_bucket_name
        self.destination_index = destination_index
        self.embedding_function = embedding_function

    def get_data_from_bucket(self) -> list[dict]:
        """
        Get all files from a 3 bucket
        """
        data = []
        s3 = boto3.client("s3")
        bucket = boto3.resource("s3").Bucket(self.source_bucket_name)
        bucket_data = bucket.objects.all()
        for object in bucket_data:
            response = s3.get_object(Bucket=self.source_bucket_name, Key=object.key)
            data.append(json.loads(response["Body"].read().decode("utf-8")))
        return data

    @abstractmethod
    def prepare_documents_for_pinecone(self) -> list[str]:
        pass

    def add_documents_from_bucket(self) -> None:
        """
        Vectorize text using the embedding function
        """
        documents = self.prepare_documents_for_pinecone()
        # create langchain documents
        # vectorize the documents
        pinecone_index = PineconeVectorStore(
            index_name=self.destination_index, embedding=self.embedding_function
        )
        pinecone_index.add_documents(documents)
        logger.info(f"Indexed {len(documents)} documents to {self.destination_index}")


class ApolloS3Pipeline(ConductorJobS3Pipeline):
    def prepare_documents_for_pinecone(self) -> list[str]:
        """Create a list of documents from the s3 data"""
        documents = []
        data = self.get_data_from_job(job_id_suffix="/apollo_person_search")
        for entry in data:
            documents.append(
                Document(
                    page_content=f"""The key player {entry['person']['name']} is the {entry['person']['title']} at {entry['person']['organization']['name']}.
                    {entry['person']['name']}'s is located in {entry['person']['city']}, {entry['person']['state']}, {entry['person']['country']}.
                    LinkedIn: {entry['person']['linkedin_url']}.
                    The engagement strategy is {entry['engagement_strategy']['strategy']}""",
                    metadata={
                        "source": "s3_apollo_person_search",
                        "job_id": self.job_id,
                        "type": "key_player_report",
                    },
                )
            )
        return documents


class DiscordS3Pipeline(ConductorJobS3Pipeline):
    def prepare_documents_for_pinecone(self) -> list[str]:
        """Create a list of documents from the s3 data"""
        data = self.get_data_from_job()
        documents = []
        for chat in data:
            if len(chat["message"]) > 0:
                documents.append(
                    Document(
                        page_content=chat["message"],
                        metadata={
                            "id": chat["id"],
                            "author": chat["author"],
                            "created_at": chat["created_at"],
                            "source": chat["source"],
                            "channel": chat["channel"],
                        },
                    )
                )
        return documents


class ApolloBulkS3Pipeline(ConductorBulkS3Pipeline):
    """
    Apollo bulk pipeline to read all data from s3 and add to Pinecone
    """

    def prepare_documents_for_pinecone(self) -> list[str]:
        documents = []
        job_id = None
        data = self.get_data_from_bucket()
        # first grab job_id
        for entry in data:
            if "input" in entry:
                job_id = entry["input"]["job_id"]
                break
        if job_id:
            for entry in data:
                if "key_players" in entry:
                    for key_player in entry["key_players"]:
                        documents.append(
                            Document(
                                page_content=f"""The key player {key_player['name']} is the {key_player['title']} at {key_player['company']['name']}.
        {key_player['company']['name']}'s background is {key_player['company']['background']}
        The engagement strategy is {key_player['strategy']}""",
                                metadata={
                                    "source": "s3_apollo_person_search",
                                    "job_id": job_id,
                                    "type": "key_player_report",
                                },
                            )
                        )
        else:
            logger.error("No job_id found in data")
            documents = None
        return documents
