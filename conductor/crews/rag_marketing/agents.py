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
            goal="Collect the urls from the research and add the vectors to a vector database.",
            backstory="An expert in researching company structure data using search engines, all available context, and tools.",
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
            goal="Collect the urls from the research and add the vectors to a vector database.",
            backstory="An expert in finding links that identify personnel using all available context and tools. Find executives, board members, and other personnel using great search engine queries.",
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
            goal="Collect the urls from the research and add the vectors to a vector database.",
            backstory="An expert in collecting data to create a list of competitors using all available context and tools. Uses the estimated size, industry, and location to find the most relevant competitors in search engines.",
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
            goal="Collect the urls from the research and add the vectors to a vector database.",
            backstory="An expert in collecting company data to create SWOT analysis using all available context and tools. Uses the estimated size, industry, and location to find the most relevant SWOT analysis in search engines.",
            tools=[
                SerpSearchEngineIngestTool(
                    elasticsearch=elasticsearch, index_name=index_name
                )
            ],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    def company_history_research_agent(
        self, elasticsearch: Elasticsearch, index_name: str, llm=None
    ) -> Agent:
        return Agent(
            role="Company History Research Agent",
            goal="Collect the urls from the research and add the vectors to a vector database.",
            backstory="An expert in collecting company data to create a company history using all available context and tools. Look at places like Wikipedia and similar websites to find the company history.",
            tools=[
                SerpSearchEngineIngestTool(
                    elasticsearch=elasticsearch, index_name=index_name
                )
            ],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    def pricing_research_agent(
        self, elasticsearch: Elasticsearch, index_name: str, llm=None
    ) -> Agent:
        return Agent(
            role="Pricing Research Agent",
            goal="Collect the urls from the research and add the vectors to a vector database.",
            backstory="An expert in collecting company data to create a pricing information in a vector database  using all available context and tools. Look at places like the company website and similar websites to find the pricing information.",
            tools=[
                SerpSearchEngineIngestTool(
                    elasticsearch=elasticsearch, index_name=index_name
                )
            ],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    def recent_events_research_agent(
        self, elasticsearch: Elasticsearch, index_name: str, llm=None
    ) -> Agent:
        return Agent(
            role="Recent Events Research Agent",
            goal="Collect the urls from the research and add the vectors to a vector database.",
            backstory="An expert in collecting company data to create data about recent events in a vector database using all available context and tools. Look at places like news websites and similar websites to find the recent events.",
            tools=[
                SerpSearchEngineIngestTool(
                    elasticsearch=elasticsearch, index_name=index_name
                )
            ],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    def product_and_services_research_agent(
        self, elasticsearch: Elasticsearch, index_name: str, llm=None
    ) -> Agent:
        return Agent(
            role="Product and Services Research Agent",
            goal="Collect the urls from the research and add the vectors to a vector database.",
            backstory="An expert in collecting company data to create a list of products and services in a vector database using all available context and tools. Look at places like the company website and similar websites to find the products and services.",
            tools=[
                SerpSearchEngineIngestTool(
                    elasticsearch=elasticsearch, index_name=index_name
                )
            ],
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    def market_research_agent(
        self, elasticsearch: Elasticsearch, index_name: str, llm=None
    ) -> Agent:
        return Agent(
            role="Market Research Agent",
            goal="Collect the urls from the research and add the vectors to a vector database.",
            backstory="An expert in collecting company data to create a market research using all available context and tools. Use industry reports and industry websites to sure up the market research.",
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
