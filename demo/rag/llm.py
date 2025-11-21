import os
from openai import OpenAI
from typing import Optional


def get_llm_client() -> OpenAI:
    """Initialize LLM client using OpenAI-compatible API."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def get_llm_answer(
    client: OpenAI, 
    context: str, 
    question: str, 
    model: str = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """
    Generate answer using OpenAI-compatible LLM based on context and question.
    
    Args:
        client: OpenAI-compatible client
        context: Retrieved context from knowledge base
        question: User's question
        model: Model name (default from environment)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
    
    Returns:
        Generated answer as string
    """
    if model is None:
        model = os.getenv("OPENAI_MODEL_NAME")
    
    # Define system and user prompts
    SYSTEM_PROMPT = """
    You are an intelligent assistant that can answer user questions based on provided context information.
    Please provide accurate and detailed answers based on the given context content.
    If there is insufficient information in the context to answer the question, please honestly indicate this.
    """
    
    USER_PROMPT = f"""
    Please answer the question based on the following context information:

    <Context>
    {context}
    </Context>

    <Question>
    {question}
    </Question>

    Please provide an accurate and helpful answer:
    """

    try:
        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            "temperature": temperature,
        }
        
        # Add max_tokens if specified
        if max_tokens:
            request_params["max_tokens"] = max_tokens

        response = client.chat.completions.create(**request_params)
        
        answer = response.choices[0].message.content
        return answer
        
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return f"Sorry, an error occurred while generating the answer: {str(e)}"


def get_llm_summary(client: OpenAI, text: str, model: str = None) -> str:
    """
    Generate a summary of the given text using OpenAI-compatible LLM.
    
    Args:
        client: OpenAI-compatible client
        text: Text to summarize
        model: Model name (default from environment)
    
    Returns:
        Summary as string
    """
    if model is None:
        model = os.getenv("OPENAI_MODEL_NAME")
    
    SYSTEM_PROMPT = """
    You are a professional text summarization assistant. Please generate concise and accurate summaries for user-provided text.
    The summary should contain the main points and key information of the text.
    """
    
    USER_PROMPT = f"""
    Please generate a summary for the following text:

    {text}

    Summary:
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=0.3,
        )
        
        summary = response.choices[0].message.content
        return summary
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Error occurred while generating summary: {str(e)}"
