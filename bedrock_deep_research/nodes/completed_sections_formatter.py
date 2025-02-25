import logging

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import Configuration
from ..model import ArticleFeedback, ArticleState, Section
from ..utils import exponential_backoff_retry

logger = logging.getLogger(__name__)


review_completed_sections_prompt = """You are an expert editor tasked to review all the sections of the article. Focus on identifying:
- Potential repetitions: If the same concept is repeated in different sections without adding value, then the subsequent sections should avoid the repetition.
- Concepts or explanations which are incomplete: if so, you must provide a feedback to cover the missing information
- Explanations which are unnecessarily overcomplicated
- Acronyms which have not been expained

This review must consider the article as a whole. First you must read all sections and then perform your analysis. Before providing any feedback, take your time to <think>.
Collect your feedbacks and make sure to relate a feedback to a specific section. Your feedback must be clear and concise so that the writer can apply corrections based on your feedbacks.

<output>
Return the list of feedbacks containing the following informations:
- Feedback: the feedback which allows the writer to correct the article
- Section_number: the number of the section this feedback relates to
- Section_Title: the title of the section this feedback relates to

If you don't have any feedback, return an empty list.
</output>

Article Title: {title}

<Sections>
{draft}
</Sections>
"""


class CompletedSectionsFormatter:
    N = "gather_completed_sections"

    def __call__(self, state: ArticleState, config: RunnableConfig):
        logger.info("Gathering completed sections")

        title = state["title"]
        completed_sections = state["completed_sections"]
        draft = self._format_sections(completed_sections)

        configurable = Configuration.from_runnable_config(config)

        writer_model = ChatBedrock(
            model_id=configurable.writer_model, streaming=True
        ).with_structured_output(ArticleFeedback)

        editor_feedback = self._generate_feedback(
            writer_model, review_completed_sections_prompt, title, draft
        )

        return {
            "report_sections_from_research": draft,
            "editor_feedback": editor_feedback,
        }

    @exponential_backoff_retry(Exception, max_retries=10)
    def _generate_feedback(
        self, model: ChatBedrock, system_prompt: str, title: str, draft: str
    ) -> ArticleFeedback:
        """Generate feedback for the completed sections"""

        # Format system instructions
        formatted_system_prompt = system_prompt.format(
            title=title, draft=draft)

        response = model.invoke(
            [SystemMessage(content=formatted_system_prompt)]
            + [HumanMessage(content="Generate a feedback on the draft of the article")]
        )

        logger.info(f"Feedback on completed sections: {response}")

        return response

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
