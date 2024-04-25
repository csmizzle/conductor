"""
Parse Apollo Person search for downstream context enhancement
"""
from conductor.models import Context
from textwrap import dedent


class ApolloPersonSearchContext(Context):
    def create_context(self, data: dict) -> list[str]:
        """
        Parse Apollo Person search for downstream context enhancement
        """
        context = []
        for entry in data:
            context_entry = dedent(
                f"""
                Name: {self.path_search("person.name", entry)}
                Organization: {self.path_search("person.organization.name", entry)}
                Organization Phone: {self.path_search("person.organization.sanitized_phone", entry)}
                Organization Founded Year: {self.path_search("person.organization.founded_year", entry)}
                Website: {self.path_search("person.organization.website_url", entry)}
                Title: {self.path_search("person.title", entry)}
                Headline: {self.path_search("person.headline", entry)}
                First Sanitized Phone: {self.path_search("person.phone_numbers[0].sanitized_number", entry)}
                Engagement Strategy: {self.path_search("engagement_strategy.strategy", entry)}
            """
            ).strip()
            context.append(context_entry)
        return context


class ApolloPersonSearchRawContext(Context):
    def create_context(self, data: dict) -> list[str]:
        """
        Parse Apollo Person search for downstream context enhancement
        """
        context = []
        for entry in data["people"]:
            context_entry = dedent(
                f"""
                Name: {self.path_search("name", entry)}
                Organization: {self.path_search("organization.name", entry)}
                Organization Phone: {self.path_search("organization.sanitized_phone", entry)}
                Organization Founded Year: {self.path_search("organization.founded_year", entry)}
                Website: {self.path_search("organization.website_url", entry)}
                Title: {self.path_search("title", entry)}
                Headline: {self.path_search("headline", entry)}
                First Sanitized Phone: {self.path_search("phone_numbers[0].sanitized_number", entry)}
            """
            ).strip()
            context.append(context_entry)
        return context
