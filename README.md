# Intelligent LLM API Router

An extensible, intent-driven API router built with FastAPI that dynamically selects the best Large Language Model (LLM) for a given query.

## Architecture & Routing Logic

The application follows a **Two-Pass Strategy**:
1. **Intent Classification:** When a request hits the endpoint, the router first sends the prompt to a highly optimized, fast model (`Groq llama3-8b-8192`) via zero-shot prompting to classify the request into one of five categories: Coding, Writing, Summarisation, Translation, or General Reasoning.
2. **Execution:** The router evaluates the classified category and dispatches the query to the most capable model for that specific vertical:
   - **Coding:** Routed to Groq (`llama3-70b-8192`) for superior logic and syntactic generation.
   - **Writing, Translation, Summarisation:** Routed to Google Gemini (`gemini-1.5-flash`) for broad contextual understanding and native multilinguality.
   - **General Reasoning:** Defaults to Gemini.

## Extensibility

The codebase utilizes the **Strategy Pattern**. Adding a new provider (e.g., OpenAI, Anthropic) requires zero modifications to the routing logic—simply implement the `LLMProvider` abstract base class in the `providers/` directory and instantiate it inside `router.py`.