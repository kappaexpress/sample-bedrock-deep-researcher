from ..model import ArticleState


def compile_final_article(state: ArticleState):
    """Compile the final report"""

    # Get sections
    title = state["title"]
    head_image_path = state["head_image_path"]
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    all_sections = f"# {title}\n\n"

    if head_image_path:
        all_sections += f"![{title}]({head_image_path})\n\n"

    # Compile final report
    all_sections += "\n\n".join([f"## {s.name}\n{s.content}" for s in sections])

    return {"final_report": all_sections, "sections": sections}
