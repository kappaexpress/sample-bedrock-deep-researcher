import logging

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import Configuration
from ..model import ArticleState, Outline

logger = logging.getLogger(__name__)

article_planner_instructions = """You are an expert technical writer tasked to plan an article outline for a topic. Your task is to generate a title and a list of sections for the article. The sections should be well-organized with clear headings that will each be individually researched.

<Section structure>
Each section should have the fields:

- Title - Title for this section of the article.
- Section Number - Incremental number used to sort the sections in the article
- Description - Brief overview of the main topics covered in this section.
- Research - Whether to perform web research for this section of the article.
- Content - The content of the section, which you will leave blank for now.

Introduction and conclusion will not require research because they can distill information from other parts of the article.
</Section structure>

<article organization>
{article_organization}
</article organization>

<Context>
Use this context to plan the sections of the article:
{context}
</Context>

{feedback}
"""


class ArticleOutlineGenerator:
    N = "generate_article_outline"

    async def __call__(self, state: ArticleState, config: RunnableConfig):
        logging.info("Generating report plan")

        topic = state.get("topic", "")
        source_str = state.get("source_str", "")
        feedback = state.get("feedback_on_report_plan", "")
        if feedback:
            feedback = f"<Feedback>\nHere is some feedback on article structure from user review:{feedback}\n</Feedback>"

        configurable = Configuration.from_runnable_config(config)

        outline = self.generate_outline(
            configurable.planner_model, article_planner_instructions, topic, configurable.report_structure, source_str, feedback)

        logger.info(f"Generated sections: {outline.sections}")

        return {"title": outline.title, "sections": outline.sections}

    def generate_outline(self, model_id: str, system_prompt: str, topic, report_structure: str, source_str: str, feedback) -> Outline:

        planner_model = ChatBedrock(
            model_id=model_id, streaming=True
        ).with_structured_output(Outline)

        system_instructions_sections = system_prompt.format(
            article_organization=report_structure,
            context=source_str,
            feedback=feedback,
        )

        return planner_model.invoke(
            [SystemMessage(content=system_instructions_sections)]
            + [
                HumanMessage(
                    content=f"Generate the sections for the topic '{topic}'. You must include 'sections' field containing a list of sections. Each section must have: title, description, plan, research, and content fields."
                )
            ]
        )
