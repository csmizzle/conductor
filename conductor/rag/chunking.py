"""
Handle large document ingestion by chunking the documents into smaller pieces and ingesting them in parallel
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from conductor.rag.models import WebPage


class WebPageContentSplitter:
    """
    Split large documents into smaller chunks
    """

    def __init__(
        self,
        webpage: WebPage,
    ) -> None:
        self.webpage = webpage
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500,
        )

    def split_content(self) -> list[str]:
        """
        Split the content into smaller chunks
        """
        return self.text_splitter.split_text(self.webpage.content)

    def create_documents(self) -> list[Document]:
        """
        Create documents from the split content
        """
        split_content = self.split_content()
        print("Split content into", len(split_content), "chunks")
        return [
            Document(
                page_content=chunk,
                metadata={
                    "url": self.webpage.url,
                    "created_at": self.webpage.created_at,
                    "raw": self.webpage.raw,
                },
            )
            for chunk in split_content
        ]


def chunk_webpage(webpage: WebPage) -> list[Document]:
    """
    Chunk the webpage content and create documents
    """
    return WebPageContentSplitter(webpage=webpage).create_documents()
