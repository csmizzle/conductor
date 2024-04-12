"""
Pinecone vector databases
- To start, let's vectorize the discord and apollo data
- Make sure to also have the data and metadata
"""
from langchain_core.embeddings.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
import boto3
import json
import logging

logger = logging.getLogger(__name__)


class ApolloS3Pipeline:
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

    def get_observation_from_job(self) -> list[dict]:
        """
        Get all files from an Apollo job
        """
        s3 = boto3.client("s3")
        bucket = boto3.resource("s3").Bucket(self.source_bucket_name)
        observation_summary = bucket.objects.filter(
            Prefix=self.job_id + "/apollo_person_search/observation/"
        )
        for object in observation_summary:
            if object.key.endswith(".json"):
                response = s3.get_object(Bucket=self.source_bucket_name, Key=object.key)
                return json.loads(response["Body"].read().decode("utf-8"))

    def get_text_from_job(self) -> list[str]:
        """
        Get all files from an Apollo job
        """
        s3 = boto3.client("s3")
        bucket = boto3.resource("s3").Bucket(self.source_bucket_name)
        observation_summary = bucket.objects.filter(
            Prefix=self.job_id + "/apollo_person_search/text/"
        )
        for object in observation_summary:
            if object.key.endswith(".txt"):
                response = s3.get_object(Bucket=self.source_bucket_name, Key=object.key)
                return response["Body"].read().decode("utf-8")

    def prepare_observation_for_pinecone(self) -> list[str]:
        """Create a list of strings from the observation"""
        documents = []
        observation = self.get_observation_from_job()
        for key_player in observation["key_players"]:
            documents.append(
                f"""The key player {key_player['name']} is the {key_player['title']} at {key_player['company']['name']}.
{key_player['company']['name']}'s background is {key_player['company']['background']}
The engagement strategy is {key_player['strategy']}"""
            )
        return documents

    def add_observations_from_job(self) -> None:
        """
        Vectorize text using the embedding function
        """

        raw_documents = self.prepare_observation_for_pinecone()
        # create langchain documents
        documents = []
        for raw_document in raw_documents:
            documents.append(
                Document(
                    page_content=raw_document,
                    metadata={
                        "source": "s3_apollo_person_search",
                        "job_id": self.job_id,
                        "type": "key_player_report",
                    },
                )
            )
        # vectorize the documents
        pinecone_index = PineconeVectorStore(
            index_name=self.destination_index, embedding=self.embedding_function
        )
        pinecone_index.add_documents(documents)
        logger.info(f"Indexed {len(documents)} documents to {self.destination_index}")


class DiscordS3Pipeline:
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

    def get_chats_from_job(self) -> list[dict]:
        """
        Get all files from an Apollo job
        """
        s3 = boto3.client("s3")
        bucket = boto3.resource("s3").Bucket(self.source_bucket_name)
        observation_summary = bucket.objects.filter(Prefix=self.job_id)
        for object in observation_summary:
            if object.key.endswith(".json"):
                response = s3.get_object(Bucket=self.source_bucket_name, Key=object.key)
                return json.loads(response["Body"].read().decode("utf-8"))

    def prepare_observation_for_pinecone(self) -> list[str]:
        """Create a list of strings from the observation"""
        documents = []
        observation = self.get_observation_from_job()
        for message in observation:
            documents.append(f"""{message['author']} said: {message['message']}""")
        return documents

    def add_observations_from_job(self) -> None:
        """
        Vectorize text using the embedding function
        """
        chats = self.get_chats_from_job()
        # create langchain documents
        documents = []
        for chat in chats:
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
        # vectorize the documents
        pinecone_index = PineconeVectorStore(
            index_name=self.destination_index, embedding=self.embedding_function
        )
        pinecone_index.add_documents(documents)
        logger.info(f"Indexed {len(documents)} documents to {self.destination_index}")
