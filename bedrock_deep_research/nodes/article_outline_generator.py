import logging

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Configuration
from ..model import ArticleState, Outline

logger = logging.getLogger(__name__)

article_planner_instructions = """You are an expert technical writer tasked to plan an article based on an input topic.

<Task>
Generate a title and a list of sections for the article.

Each section should have the fields:

- Title - Title for this section of the article.
- Section Number - Incremental number used to sort the sections in the article
- Description - Brief overview of the main topics covered in this section.
- Research - Whether to perform web research for this section of the article.
- Content - The content of the section, which you will leave blank for now.

For example, introduction and conclusion will not require research because they will distill information from other parts of the article.
</Task>

<Topic>
The topic of the article is:
{topic}
</Topic>

<article organization>
The article should follow this organization:
{article_organization}
</article organization>

<Context>
Here is context to use to plan the sections of the article:
{context}
</Context>

<Feedback>
Here is feedback on the article structure from review (if any):
{feedback}
</Feedback>"""


class ArticleOutlineGenerator:
    N = "generate_article_outline"

    async def __call__(self, state: ArticleState, config: Configuration):
        logging.info("Generating report plan")

        topic = state["topic"]
        source_str = state["source_str"]
        feedback = state.get("feedback_on_report_plan", None)

        configurable = Configuration.from_runnable_config(config)

        logger.info(f"Using Configuration: {configurable}")

        report_structure = configurable.report_structure

        planner_model = ChatBedrock(
            model_id=configurable.planner_model, streaming=True
        ).with_structured_output(Outline)

        # Format system instructions
        system_instructions_sections = article_planner_instructions.format(
            topic=topic,
            article_organization=report_structure,
            context=source_str,
            feedback=feedback,
        )

        # Generate sections
        outline = planner_model.invoke(
            [SystemMessage(content=system_instructions_sections)]
            + [
                HumanMessage(
                    content="Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. Each section must have: title, description, plan, research, and content fields."
                )
            ]
        )

        title = outline.title
        sections = outline.sections

        logger.info(f"Generated sections: {sections}")

        return {"title": title, "sections": sections}
