import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .config import Configuration
from .model import (
    ArticleInputState,
    ArticleOutputState,
    ArticleState,
    SectionOutputState,
    SectionState,
)
from .nodes import (
    ArticleHeadImageGenerator,
    ArticleOutlineGenerator,
    CompletedSectionsFormatter,
    FinalSectionsWriter,
    HumanFeedbackProvider,
    InitialResearcher,
    SectionSearchQueryGenerator,
    SectionWebResearcher,
    SectionWriter,
    compile_final_article,
    initiate_final_section_writing,
)
from .web_search import WebSearch

logger = logging.getLogger(__name__)


class BedrockDeepResearch:
    def __init__(self, config: dict, tavily_api_key: str):
        self.config = config
        self.web_search = WebSearch(tavily_api_key, save_search_results=True)
        self.graph = self.__create_workflow()

    def __create_workflow(self):
        builder = StateGraph(
            ArticleState,
            input=ArticleInputState,
            output=ArticleOutputState,
            config_schema=Configuration,
        )

        # Add nodes
        section_builder = StateGraph(SectionState, output=SectionOutputState)
        section_builder.add_node(
            SectionSearchQueryGenerator.N, SectionSearchQueryGenerator()
        )
        section_builder.add_node(
            SectionWebResearcher.N, SectionWebResearcher(self.web_search)
        )
        section_builder.add_node(SectionWriter.N, SectionWriter())

        # Add edges
        section_builder.add_edge(START, SectionSearchQueryGenerator.N)
        section_builder.add_edge(SectionSearchQueryGenerator.N, SectionWebResearcher.N)
        section_builder.add_edge(SectionWebResearcher.N, SectionWriter.N)

        builder.add_node(InitialResearcher.N, InitialResearcher(self.web_search))
        builder.add_node(ArticleOutlineGenerator.N, ArticleOutlineGenerator())
        builder.add_node(HumanFeedbackProvider.N, HumanFeedbackProvider())
        builder.add_node("build_section_with_web_research", section_builder.compile())
        builder.add_node(CompletedSectionsFormatter.N, CompletedSectionsFormatter())
        builder.add_node(FinalSectionsWriter.N, FinalSectionsWriter())
        builder.add_node(ArticleHeadImageGenerator.N, ArticleHeadImageGenerator())
        builder.add_node("compile_final_article", compile_final_article)

        builder.add_edge(START, InitialResearcher.N)
        builder.add_edge(InitialResearcher.N, ArticleOutlineGenerator.N)
        builder.add_edge(ArticleOutlineGenerator.N, HumanFeedbackProvider.N)
        builder.add_edge(
            "build_section_with_web_research", CompletedSectionsFormatter.N
        )
        builder.add_conditional_edges(
            CompletedSectionsFormatter.N,
            initiate_final_section_writing,
            [FinalSectionsWriter.N],
        )
        builder.add_edge(FinalSectionsWriter.N, ArticleHeadImageGenerator.N)
        builder.add_edge(ArticleHeadImageGenerator.N, "compile_final_article")
        builder.add_edge("compile_final_article", END)

        memory = MemorySaver()

        return builder.compile(checkpointer=memory)

    async def start(self, topic: str):
        """Starts the workflow with the given topic."""

        logger.debug(f"Starting workflow with topic: {topic}")

        return await self.graph.ainvoke(
            {"topic": topic}, self.config, stream_mode="updates"
        )

    async def feedback(self, feedback):
        """Provides feedback to the workflow."""

        logger.info(f"Feedback received: {feedback}")

        return await self.graph.ainvoke(
            Command(resume=feedback), self.config, stream_mode="updates"
        )

    def get_state(self):
        """Returns the current state of the workflow."""

        return self.graph.get_state(self.config)
