"""
First task is to collect data from a website and ingest it into a vector database.
Second task is to search the vector database for relevant information.
"""
from crewai import Task, Agent
from textwrap import dedent
from conductor.reports.models import ReportStyle
from conductor.crews.marketing.utils import write_report_prompt
from conductor.models import NamedTask


class RagMarketingTasks:
    def url_collection_task(
        self, name: str, agent: Agent, company_url: str
    ) -> NamedTask:
        return NamedTask(
            name=name,
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

    def vector_metadata_search_task(
        self, name: str, agent: Agent, url: str
    ) -> NamedTask:
        return NamedTask(
            name=name,
            description=dedent(
                f"""
            Determine what company the URL belongs to by searching the vector database.
            URL to find the website: {url}.
            Include all source links in the response.
            """
            ),
            agent=agent,
            expected_output="Confirmation that the data has been collected and ingested into the vector database or already exists.",
        )

    def swot_research_task(
        self, name: str, agent: Agent, context: list[Task] = None
    ) -> NamedTask:
        return NamedTask(
            name=name,
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
        self, name: str, agent: Agent, context: list[Task] = None
    ) -> NamedTask:
        return NamedTask(
            name=name,
            description=dedent(
                """
            Find the company structure of a company using search engine search techniques.
            Use 5 different search queries to find the most relevant information.
            Use the provided context to determine the best approach to find the information.
            The output should be a simple confirmation that the data has been collected and ingested into the vector database or already exists.
            """
            ),
            agent=agent,
            expected_output="Confirmation that the data has been collected and ingested into the vector database or already exists.",
            context=context,
        )

    def personnel_research_task(
        self, name: str, agent: Agent, context: list[Task] = None
    ) -> NamedTask:
        return NamedTask(
            name=name,
            description=dedent(
                """
            Find the personnel of a company using search engine search techniques.
            Use 5 different search queries to find the most relevant information.
            Use the provided context to determine the best approach to find the information.
            The output should be a simple confirmation that the data has been collected and ingested into the vector database or already exists.
            """
            ),
            agent=agent,
            expected_output="Confirmation that the data has been collected and ingested into the vector database or already exists.",
            context=context,
        )

    def competitor_research_task(
        self, name: str, agent: Agent, context: list[Task] = None
    ) -> NamedTask:
        return NamedTask(
            name=name,
            description=dedent(
                """
            Find information about a company's competitors using search engine search techniques.
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
        self,
        name: str,
        agent: Agent,
        search_query: str,
        context: list[Task] = None,
        instructions: str = None,
    ) -> NamedTask:
        return NamedTask(
            name=name,
            description=dedent(
                f"""
            Search the vector database for relevant information using the search query and context provided.
            Always use the specific company name(s), markets, or a combination of them in your search, dont refer to it as "the company" or use general search queries unless it makes sense.
            Search query: {search_query}.
            If there are additional instructions, follow them to find the most relevant information.
            The search queries should be at least 15 words long, and should be different from each other.
            Instructions: {instructions if instructions else 'None'}.
            Include all source links in the response.
            """
            ),
            agent=agent,
            expected_output="A comprehensive answer to the search query.",
            context=context,
        )

    def vector_multi_search_task(
        self,
        name: str,
        agent: Agent,
        search_query: str,
        context: list[Task],
        instructions: str = None,
    ) -> NamedTask:
        return NamedTask(
            name=name,
            description=dedent(
                f"""
            Search the vector database for relevant information using the search query and context provided.
            Always use the specific company name(s), markets, or a combination of them in your search, dont refer to it as "the company" or use general search queries unless it makes sense.
            Search query: {search_query}.
            Look at the search query to generate 5 other queriers that will increase the chances of finding the most relevant information.
            The search queries should be at least 15 words long, and should be different from each other.
            If there are additional instructions, follow them to find the most relevant information.
            Instructions: {instructions if instructions else 'None'}.
            Include all source links in the response.
            """
            ),
            agent=agent,
            expected_output="A comprehensive answer to the search query.",
            context=context,
        )

    def company_history_research_task(
        self, name: str, agent: Agent, context: list[Task] = None
    ) -> NamedTask:
        return NamedTask(
            name=name,
            description=dedent(
                """
            Find the history of a company using search engine search techniques.
            Use 5 different search queries to find the most relevant information.
            Use the provided context to determine the best approach to find the information.
            The output should be a simple confirmation that the data has been collected and ingested into the vector database or already exists.
            """
            ),
            agent=agent,
            expected_output="Confirmation that the data has been collected and ingested into the vector database or already exists.",
            context=context,
        )

    def pricing_research_task(
        self, name: str, agent: Agent, context: list[Task] = None
    ) -> NamedTask:
        return NamedTask(
            name=name,
            description=dedent(
                """
            Find pricing information of a company using search engine search techniques.
            Use 5 different search queries to find the most relevant information.
            Use the provided context to determine the best approach to find the information.
            The output should be a simple confirmation that the data has been collected and ingested into the vector database or already exists.
            """
            ),
            agent=agent,
            expected_output="Confirmation that the data has been collected and ingested into the vector database or already exists.",
            context=context,
        )

    def recent_events_research_task(
        self, name: str, agent: Agent, context: list[Task] = None
    ) -> NamedTask:
        return NamedTask(
            name=name,
            description=dedent(
                """
            Find the most recent events of a company using search engine search techniques.
            Use 5 different search queries to find the most relevant information.
            Use the provided context to determine the best approach to find the information.
            The output should be a simple confirmation that the data has been collected and ingested into the vector database or already exists.
            """
            ),
            agent=agent,
            expected_output="Confirmation that the data has been collected and ingested into the vector database or already exists.",
            context=context,
        )

    def products_and_services_research_task(
        self, name: str, agent: Agent, context: list[Task] = None
    ) -> NamedTask:
        return NamedTask(
            name=name,
            description=dedent(
                """
            Find key products and services of a company using search engine search techniques.
            Use 5 different search queries to find the most relevant information.
            Use the provided context to determine the best approach to find the information.
            The output should be a simple confirmation that the data has been collected and ingested into the vector database or already exists.
            """
            ),
            agent=agent,
            expected_output="Confirmation that the data has been collected and ingested into the vector database or already exists.",
            context=context,
        )

    def market_analysis_research_task(
        self, name: str, agent: Agent, context: list[Task] = None
    ) -> NamedTask:
        return NamedTask(
            name=name,
            description=dedent(
                """
            Determine which market the company operates in and what their TAM/SAM/SOM is.
            Find estimates if exact numbers are not available.
            Use the provided context to determine the best approach to find the information.
            The output should be a simple confirmation that the data has been collected and ingested into the vector database or already exists.
            """
            ),
            agent=agent,
            expected_output="Confirmation that the data has been collected and ingested into the vector database or already exists.",
            context=context,
        )

    def company_report_task(
        self,
        name: str,
        agent: Agent,
        context: list[Task],
        report_style: ReportStyle,
        key_questions: list[str] = None,
    ) -> NamedTask:
        return NamedTask(
            name=name,
            description=write_report_prompt(
                report_style=report_style,
                key_questions=key_questions,
            ),
            agent=agent,
            context=context,
            expected_output="Comprehensive report on the company with the sections overview, swot analysis, and competitors.",
        )
