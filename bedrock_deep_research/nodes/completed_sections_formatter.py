import logging

from langchain_core.runnables import RunnableConfig

from ..model import ArticleState, Section

logger = logging.getLogger(__name__)


class CompletedSectionsFormatter:
    N = "gather_completed_sections"

    def __call__(self, state: ArticleState, config: RunnableConfig):
        logger.info("Gathering completed sections")

        completed_sections = state["completed_sections"]
        draft = self._format_sections(completed_sections)

        return {
            "report_sections_from_research": draft,
        }

    def _format_sections(self, sections: list[Section]) -> str:
        """Format a list of sections into a string"""
        formatted_str = ""
        for idx, section in enumerate(sections, 1):
            formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research:
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
        return formatted_str
