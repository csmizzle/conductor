"""
Rag Marketing Agents
Key difference between Marketing and Rag Marketing is
going to first utilize vectorized data querying and collection rather than simple passing
data directly to the model.
"""
from crewai.agent import Agent
from conductor.crews.rag_marketing.tools import (
    ScrapeWebsiteIngestTool,
    VectorSearchTool,
    VectorSearchMetaTool,
    SerpSearchEngineIngestTool,
)
from elasticsearch import Elasticsearch


class MarketingRagAgents:
    """
    Create Marketing Rag Agents
    """

    def data_collection_agent(
        self, elasticsearch: Elasticsearch, index_name: str, llm=None
    ) -> Agent:
        """
        Agent that collects data from a vector database
        """
        return Agent(
            role="Data Collection Agent",
            goal="Collect urls and add the vectors to a vector database.",
            backstory="Expert at collecting data from a vector database. Uses the data to find the most relevant information.",
            tools=[
                ScrapeWebsiteIngestTool(
                    elasticsearch=elasticsearch, index_name=index_name
                )
            ],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    def company_structure_research_agent(
        self, elasticsearch: Elasticsearch, index_name: str, llm=None
    ) -> Agent:
        return Agent(
            role="Company Structure Research Agent",
            goal="Find the company structure of a company using search engine search techniques.",
            backstory="An expert in analyzing company data to create a company structure using all available context and tools. Look at places like OpenGov and similar websites to find the company structure of a company.",
            tools=[
                SerpSearchEngineIngestTool(
                    elasticsearch=elasticsearch, index_name=index_name
                )
            ],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    def personnel_research_agent(
        self, elasticsearch: Elasticsearch, index_name: str, llm=None
    ) -> Agent:
        return Agent(
            role="Personnel Research Agent",
            goal="Find the personnel of a company using search engine search techniques.",
            backstory="An expert in analyzing company data to create a list of personnel using all available context and tools. Find executives, board members, and other personnel.",
            tools=[
                SerpSearchEngineIngestTool(
                    elasticsearch=elasticsearch, index_name=index_name
                )
            ],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    def competitor_research_agent(
        self, elasticsearch: Elasticsearch, index_name: str, llm=None
    ) -> Agent:
        return Agent(
            role="Competitor Research Agent",
            goal="Find information about a company's competitors.",
            backstory="An expert in analyzing company data to create a list of competitors using all available context and tools. Uses the estimated size, industry, and location to find the most relevant competitors.",
            tools=[
                SerpSearchEngineIngestTool(
                    elasticsearch=elasticsearch, index_name=index_name
                )
            ],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    def swot_research_agent(
        self, elasticsearch: Elasticsearch, index_name: str, llm=None
    ) -> Agent:
        """
        Collect and vectorize data for a SWOT analysis
        """
        return Agent(
            role="SWOT Research Agent",
            goal="Extensively use search engine search techniques to find relevant information for a SWOT analysis for a company.",
            backstory="An expert in analyzing company data to create SWOT analysis using all available context and tools.",
            tools=[
                SerpSearchEngineIngestTool(
                    elasticsearch=elasticsearch, index_name=index_name
                )
            ],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    def vector_search_agent(
        self, elasticsearch: Elasticsearch, index_name: str, llm=None
    ) -> Agent:
        """
        Agent that searches a vector database
        """
        return Agent(
            role="Vector Search Agent",
            goal="Find relevant information from a vector database using queries.",
            backstory="Expert at finding using vector search techniques. Create several queries to find the most relevant information.",
            tools=[
                VectorSearchTool(elasticsearch=elasticsearch, index_name=index_name)
            ],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    def vector_search_metadata_agent(
        self, elasticsearch: Elasticsearch, index_name: str, llm=None
    ) -> Agent:
        """
        Agent that searches a vector database for metadata
        """
        return Agent(
            role="Vector Search Metadata Agent",
            goal="Find metadata from a vector database using queries.",
            backstory="Expert at finding using vector search techniques. Create several queries to find the most relevant metadata.",
            tools=[
                VectorSearchMetaTool(
                    elasticsearch=elasticsearch,
                    index_name=index_name,
                )
            ],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )
