"""
First task is to collect data from a website and ingest it into a vector database.
Second task is to search the vector database for relevant information.
"""
from crewai import Task, Agent
from textwrap import dedent


class RagMarketingTasks:
    def url_collection_task(self, agent: Agent, company_url: str) -> Task:
        return Task(
            description=dedent(
                f"""
            Collect data from a website and ingest it into a vector database.
            URL to find the website: {company_url}.
            The output should be a simple confirmation that the data has been collected and ingested into the vector database or already exists.
            """
            ),
            agent=agent,
            expected_output="Confirmation that the data has been collected and ingested into the vector database or already exists.",
        )

    def vector_metadata_search_task(self, agent: Agent, url: str, query: str) -> Task:
        return Task(
            description=dedent(
                f"""
            Determine what company the URL belongs to by searching the vector database.
            URL to find the website: {url}.
            """
            ),
            agent=agent,
            expected_output="Confirmation that the data has been collected and ingested into the vector database or already exists.",
        )

    def swot_research_task(self, agent: Agent, context: list[Task] = None) -> Task:
        return Task(
            description=dedent(
                """
            Extensively use search engine search techniques to find relevant information for a SWOT analysis for a company.
            Use 5 different search queries to find the most relevant information.
            Use the provided context to determine the best approach to find the information.
            The output should be a simple confirmation that the data has been collected and ingested into the vector database or already exists.
            """
            ),
            agent=agent,
            expected_output="Confirmation that the data has been collected and ingested into the vector database or already exists.",
            context=context,
        )

    def company_structure_research_task(
        self, agent: Agent, context: list[Task] = None
    ) -> Task:
        return Task(
            description=dedent(
                """
            Find the company structure of a company using search engine search techniques.
            Check places like OpenGov and similar websites to find the company structure of a company.
            Use 5 different search queries to find the most relevant information.
            Use the provided context to determine the best approach to find the information.
            The output should be a simple confirmation that the data has been collected and ingested into the vector database or already exists.
            """
            ),
            agent=agent,
            expected_output="Confirmation that the data has been collected and ingested into the vector database or already exists.",
            context=context,
        )

    def vector_search_task(
        self, agent: Agent, search_query: str, context: list[Task] = None
    ) -> Task:
        return Task(
            description=dedent(
                f"""
            Find relevant information from a vector database using queries.
            Search query: {search_query}.
            """
            ),
            agent=agent,
            expected_output="A comprehensive answer to the search query.",
            context=context,
        )

    def vector_multi_search_task(
        self, agent: Agent, search_query: str, context: list[Task]
    ) -> Task:
        return Task(
            description=dedent(
                f"""
            Search the vector database for relevant information using the search query and context provided.
            Search query: {search_query}.
            Look at the search query to generate 5 other queriers that will increase the chances of finding the most relevant information.
            """
            ),
            agent=agent,
            expected_output="A comprehensive answer to the search query.",
            context=context,
        )
