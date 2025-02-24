import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import List

import nest_asyncio
import pyperclip
import pytz
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from bedrock_deep_research import BedrockDeepResearch
from bedrock_deep_research.model import Section

DEFAULT_MODEL_ID = "us.anthropic.claude-3-haiku-20240307-v1:0"
DEFAULT_NUM_SEARCH_QUERIES = 2
DEFAULT_MAX_SEARCH_DEPTH = 2
DEFAULT_TOPIC = "Upload files using Amazon S3 presigned url in Python"
DEFAULT_WRITING_GUIDELINES = """- Strict 150-200 word limit
- Start with your most important insight in **bold**
"""

SUPPORTED_MODELS = {
    "Anthropic Claude 3.5 Haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "Anthropic Claude 3.5 Sonnet v2": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "Amazon Nova Lite": "amazon.nova-lite-v1:0",
    "Amazon Nova Pro": "amazon.nova-pro-v1:0",
}

logger = logging.getLogger(__name__)

nest_asyncio.apply()


class Article(BaseModel):
    title: str = Field(description="Title of the article")
    date: str = Field(
        description="Date of the article",
        default=datetime.now(pytz.UTC).strftime("%Y-%m-%d"),
    )
    sections: List[Section] = Field(description="List of sections in the article")

    def render_outline(self) -> str:
        sections_content = "\n".join(
            f"{i+1}. {section.name}" for i, section in enumerate(self.sections)
        )
        return f"""
Title: {self.title}

{sections_content}
"""

    def render_section(self, section: Section) -> str:
        section_str = f"""
## {section.name}

{section.content}
"""

        return section_str

    def render_full_article(self) -> str:
        sections_content = "\n".join(
            self.render_section(section) for section in self.sections
        )

        # references = "\n".join(
        #     f"- {ref}" for section in self.sections for ref in section.references
        # )

        return f"""# {self.title}
#### Date: {self.date}

{sections_content}
"""


def reset_state():
    if "head_image_path" not in st.session_state:
        st.session_state.head_image_path = None
    if "bedrock_deep_research" not in st.session_state:
        st.session_state.bedrock_deep_research = None
    if "stage" not in st.session_state:
        st.session_state.stage = "initial_form"
    if "article" not in st.session_state:
        st.session_state.article = ""
    if "text_error" not in st.session_state:
        st.session_state.text_error = ""
    if "accept_draft" not in st.session_state:
        st.session_state.accept_draft = False
    if "cb_handler" not in st.session_state:
        st.session_state.cb_handler = None
    if "task" not in st.session_state:
        st.session_state.task = None


def render_initial_form():
    """
    Renders the initial form for article generation with topic and writing_guidelines inputs.
    """
    loop = None
    try:
        loop = asyncio.get_event_loop()

        with st.form("article_form"):
            topic = st.text_area(
                "Topic",
                value=DEFAULT_TOPIC,
                help="Enter the topic you want to write about",
            )

            writing_guidelines = st.text_area(
                "Writing Guidelines",
                value=DEFAULT_WRITING_GUIDELINES,
                help="Enter any specific guidelines regarding the writing length and style",
            )

            planner_model_name = st.selectbox(
                "Select your Model for planning tasks", SUPPORTED_MODELS.keys()
            )

            writer_model_name = st.selectbox(
                "Select your Model for writing tasks", SUPPORTED_MODELS.keys()
            )

            number_of_queries = st.number_input(
                "Number of queries generated for each web search",
                min_value=1,
                max_value=10,
                value=DEFAULT_NUM_SEARCH_QUERIES,
            )

            max_search_depth = st.number_input(
                "Maximum number of reflection and web search iterations allowed for each sections",
                min_value=1,
                max_value=10,
                value=DEFAULT_MAX_SEARCH_DEPTH,
            )

            submitted = st.form_submit_button("Generate Outline", type="primary")

            if submitted:
                logger.info(
                    f"generate_article on '{topic}' following '{writing_guidelines}'"
                )

                # logger.info(f"Using model: {model_id}")
                if not topic:
                    st.session_state.text_error = "Please enter a topic"
                    return

                if not writing_guidelines:
                    st.session_state.text_error = (
                        "Please enter your writing guidelines for the article"
                    )
                    return

                config = {
                    "configurable": {
                        "thread_id": str(uuid.uuid4()),
                        "writing_guidelines": writing_guidelines,
                        "max_search_depth": max_search_depth,
                        "number_of_queries": number_of_queries,
                        "planner_model": SUPPORTED_MODELS.get(planner_model_name),
                        "writer_model": SUPPORTED_MODELS.get(writer_model_name),
                    }
                }

                st.session_state.bedrock_deep_research = BedrockDeepResearch(
                    config=config, tavily_api_key=os.getenv("TAVILY_API_KEY")
                )

                with st.session_state.text_spinner_placeholder:
                    with st.spinner(
                        "Please wait while the article outline is being generated..."
                    ):
                        # response = await st.session_state.bedrock_deep_research.start(topic)
                        response = loop.run_until_complete(
                            st.session_state.bedrock_deep_research.start(topic)
                        )
                        logger.info(f"Outline response: {response}")
                        state = st.session_state.bedrock_deep_research.get_state()

                        article = Article(
                            title=state.values["title"],
                            sections=state.values["sections"],
                        )
                        st.session_state.article = article.render_outline()
                        st.session_state.stage = "outline_feedback"
                        st.rerun()
    # except Exception as e:
    #     logger.error(f"An error occurred: {e}")
    #     raise
    finally:
        # Clean up the event loop if it was created
        if loop and not loop.is_closed():
            loop.close()


def render_outline_feedback(article_container):
    """
    Renders the article outline and gets user feedback.
    """
    with article_container.container():
        st.markdown("## Article Outline")
        st.markdown(st.session_state.article)

    st.markdown("Please provide feedback to the outline")

    with st.form("feedback_form"):

        feedback = st.text_area(
            label="Feedback",
            placeholder="Input a feedback to change the title or the sections",
        )

        col1, col2 = st.columns(2)

        with col1:
            submit_feedback_pressed = st.form_submit_button("Submit Feedback")

            if submit_feedback_pressed:
                on_submit_button_click(feedback)

        with col2:
            accept_outline_pressed = st.form_submit_button(
                "Accept Outline", type="primary"
            )

            if accept_outline_pressed:
                on_accept_outline_button_click()


def render_final_result(article_container):
    """
    Renders the final article with options to copy or start over.
    """

    with article_container.container():

        st.image(st.session_state.head_image_path, width=1200)
        st.markdown(st.session_state.article)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Copy to Clipboard", type="primary"):
            st.write("Article copied to clipboard!")
            st.toast("Article copied to clipboard!")
            pyperclip.copy(st.session_state.article)

    with col2:
        if st.button("Start Over"):
            reset_state()
            st.rerun()

    # Display any error messages
    if st.session_state.text_error:
        st.error(st.session_state.text_error)


def on_submit_button_click(feedback):
    loop = None
    try:
        loop = asyncio.get_event_loop()
        logger.info("Submit feedback pressed")
        if not feedback:
            st.session_state.text_error = "Please enter a feedback"
            return

        with st.session_state.text_spinner_placeholder:
            with st.spinner("Please wait while your feedback is being processed"):
                try:
                    response = loop.run_until_complete(
                        st.session_state.bedrock_deep_research.feedback(feedback)
                    )

                    logger.info(f"Feedback response: {response}")

                    state = st.session_state.bedrock_deep_research.get_state()

                    article = Article(
                        title=state.values["title"], sections=state.values["sections"]
                    )

                    st.session_state.article = article.render_outline()
                    st.session_state.text_error = ""
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        st.error(f"An error occurred: {e}")
    finally:
        if loop and not loop.is_closed():
            loop.close()


def on_accept_outline_button_click():
    logger.info("Accept outline pressed")

    loop = None
    try:
        loop = asyncio.get_event_loop()
        # if st.form_submit_button("Accept Outline", type="primary"):
        with st.session_state.text_spinner_placeholder:
            with st.spinner("Please wait while the article is being generated..."):
                try:
                    response = loop.run_until_complete(
                        st.session_state.bedrock_deep_research.feedback(True)
                    )

                    logger.info(f"Accept outline response: {response}")

                    state = st.session_state.bedrock_deep_research.get_state()

                    article = Article(
                        title=state.values["title"], sections=state.values["sections"]
                    )

                    st.session_state.head_image_path = state.values["head_image_path"]
                    st.session_state.article = state.values["final_report"]

                    st.session_state.stage = "final_result"
                    st.session_state.text_error = ""
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        st.error(f"An error occurred: {e}")
    finally:
        if loop and not loop.is_closed():
            loop.close()


def main():
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()

    logging.basicConfig(
        level=LOGLEVEL,
        force=True,
        format="%(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(message)s",
    )
    load_dotenv()

    reset_state()

    title_container = st.container()
    col1, col2 = st.columns([1, 5])
    with title_container:
        with col1:
            st.image("static/bedrock-icon.png", width=100)
        with col2:
            st.title("Bedrock Deep Research")

    st.divider()

    st.session_state.text_spinner_placeholder = st.empty()
    article_placeholder = st.empty()

    # st.session_state.cb_handler = get_streamlit_cb(
    #     article_placeholder.container())

    if st.session_state.stage == "initial_form":
        render_initial_form()
    elif st.session_state.stage == "outline_feedback":
        render_outline_feedback(article_placeholder)
    elif st.session_state.stage == "final_result":
        render_final_result(article_placeholder)


if __name__ == "__main__":
    main()
