"""
Create context from Apify_ data
"""
from conductor.models import Context
from textwrap import dedent


class ApifySummaryContext(Context):
    """
    Parse Apify summary for downstream context enhancement
    """

    def create_context(self, data: dict) -> list[str]:
        """Parse raw Apify summary for downstream context enhancement

        Args:
            data (dict): dictionary with url, raw, and summary keys

        Returns:
            list[str]: list of context entries
        """
        context = []
        for entry in data:
            context_entry = dedent(
                f"""
                Competitor URL: {entry['url']}
                Competitor Summary: {entry['summary']}
            """
            ).strip()
            context.append(context_entry)
        return context
