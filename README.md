# AI-Powered Article Generation with Bedrock Deep Research

Agentic Writer is a Streamlit-based application that automates article creation through AI-powered research, writing, and image generation. It combines web research, structured content generation, and human feedback to produce comprehensive, well-researched articles with accompanying header images.

The application streamlines the article creation process by breaking it down into manageable steps: initial research, outline generation, section writing with web research integration, and final compilation. It leverages advanced language models like Claude 3 Haiku for content generation and includes features for human feedback and revision at key stages. The system maintains writing quality through configurable guidelines and iterative refinement based on user input.

## Repository Structure
```
bedrock_deep_research/
├── bedrock_deep_research.py          # Main Streamlit application entry point
├── bedrock_deep_research/
│   ├── config.py             # Configuration settings and parameters
│   ├── graph.py              # Core workflow orchestration using LangGraph
│   ├── model.py              # Data models for articles and sections
│   ├── nodes/                # Individual workflow components
│   │   ├── article_head_image_generator.py    # Header image generation
│   │   ├── article_outline_generator.py       # Article outline creation
│   │   ├── section_writer.py                  # Section content generation
│   │   └── [other node files]                 # Additional workflow components
│   ├── utils.py              # Utility functions
│   └── web_search.py         # Web research integration using Tavily API
├── poetry.lock               # Poetry dependency lock file
└── pyproject.toml           # Project configuration and dependencies
```

## Usage Instructions

### Prerequisites
- Python 3.8+
- Poetry for dependency management
- Tavily API key for web research
- AWS Bedrock access for language model inference
- Environment variables configured in `.env`:
  - `TAVILY_API_KEY`
  - AWS credentials

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd bedrock-deep-research

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Quick Start
1. Start the Streamlit application:
```bash
streamlit run bedrock_deep_research.py
```

2. In the web interface:
   - Enter your article topic
   - Specify writing guidelines
   - Configure search parameters
   - Click "Generate Outline"

3. Review and provide feedback on the outline:
   - Submit feedback to refine the outline
   - Accept the outline to proceed with full article generation

4. Once complete:
   - Review the generated article with header image
   - Copy to clipboard or start a new article

### More Detailed Examples

**Custom Writing Guidelines Example:**
```text
- Strict 150-200 word limit
- Start with your most important insight in **bold**
- Include code examples where relevant
- Focus on practical implementation
```

**Web Research Configuration:**
```python
number_of_queries = 3  # Number of search queries per section
max_search_depth = 2   # Maximum research iterations per section
```

### Troubleshooting

**Common Issues:**

1. API Authentication Errors
   - Error: "Invalid API credentials"
   - Solution: Verify Tavily API key in `.env` file
   - Check AWS credentials configuration

2. Content Generation Timeout
   - Error: "Request timed out"
   - Increase timeout settings in `config.py`
   - Reduce number of concurrent requests

3. Web Research Failures
   - Check internet connectivity
   - Verify Tavily API rate limits
   - Review search query formatting in logs

**Debug Mode:**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
streamlit run bedrock_deep_research.py
```

## Data Flow
The application follows a sequential workflow from topic input to final article generation, with feedback loops for refinement.

```ascii
[Topic Input] -> [Initial Research] -> [Outline Generation] -> [Human Feedback]
                      |                       |                      |
                      v                       v                      v
              [Web Research] <- [Section Writing] <- [Outline Refinement]
                      |                |                     |
                      v                v                     v
              [Content Review] -> [Final Assembly] -> [Image Generation]
```

Key Component Interactions:
- InitialResearcher performs web searches to gather context
- ArticleOutlineGenerator creates structured outline using research data
- HumanFeedbackProvider enables interactive refinement
- SectionWriter generates content with web research integration
- ArticleHeadImageGenerator creates relevant header images
- Final compilation combines all elements into cohesive article