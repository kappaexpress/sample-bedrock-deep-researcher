
import logging
import random
import time
from functools import wraps

logger = logging.getLogger(__name__)


def exponential_backoff_retry(
    ExceptionToCheck, max_retries: int = 5, initial_delay: float = 1.0
):
    """
    Decorator that implements exponential backoff retry logic.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except ExceptionToCheck as e:

                    if attempt == max_retries:
                        logger.error(
                            f"Execution failed after {attempt} attempts")
                        raise e

                    # Add jitter to avoid thundering herd problem
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = delay + jitter

                    logger.debug(
                        f"Attempt {attempt + 1}/{max_retries} failed. {str(e)}"
                        f"Retrying in {sleep_time:.2f} seconds..."
                    )

                    time.sleep(sleep_time)
                    delay *= 2  # Exponential backoff

        return wrapper

    return decorator


def format_web_search(search_response, max_tokens_per_source, include_raw_content=True):
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(search_response, 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(
                    f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()
