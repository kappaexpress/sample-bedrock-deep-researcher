from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import Configuration
from ..model import Queries, SectionState

# Query writer instructions
query_writer_instructions = """You are an expert technical writer crafting targeted web search queries that will gather comprehensive information for writing a technical article section.

<Section topic>
{section_topic}
</Section topic>


<Task>
When generating {number_of_queries} search queries, ensure they:
1. Cover different aspects of the topic (e.g., core features, real-world applications, technical architecture)
2. Include specific technical terms related to the topic
3. Target recent information by including year markers where relevant (e.g., "2024")
4. Look for comparisons or differentiators from similar technologies/approaches
5. Search for both official documentation and practical implementation examples

Your queries should be:
- Specific enough to avoid generic results
- Technical enough to capture detailed implementation information
- Diverse enough to cover all aspects of the section plan
- Focused on authoritative sources (documentation, technical blogs, academic papers)
</Task>"""


class SectionSearchQueryGenerator:
    N = "generate_section_search_queries"

    def __call__(self, state: SectionState, config: RunnableConfig):
        """Generate search queries for a article section"""

        # Get state
        section = state["section"]

        # Get configuration
        configurable = Configuration.from_runnable_config(config)

        planner_model = ChatBedrock(
            model_id=configurable.planner_model
        ).with_structured_output(Queries)

        # Format system instructions
        system_instructions = query_writer_instructions.format(
            section_topic=section.description, number_of_queries=configurable.number_of_queries
        )

        # Generate queries
        queries = planner_model.invoke(
            [SystemMessage(content=system_instructions)]
            + [HumanMessage(content="Generate search queries on the provided topic.")]
        )

        return {"search_queries": queries.queries}
