import streamlit as st
import google.generativeai as genai
from groq import Groq

# --- App Configuration ---
st.set_page_config(page_title="Intelligent LLM Router", page_icon="🧠", layout="centered")
st.title("🧠 Intelligent LLM Router")
st.write("Dynamically routing your queries to the best model.")

try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    groq_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("API keys are missing! Please configure them in Streamlit Secrets.")
    st.stop()

# --- Initialize Clients ---
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
groq_client = Groq(api_key=groq_key)

# --- Routing Logic ---
def classify_intent(query: str) -> str:
    prompt = f"""Classify the following query into exactly one of these categories:
    - Coding
    - Writing
    - Summarisation
    - Translation
    - General reasoning
    
    Respond with ONLY the exact category name. Do not include any other text.
    
    Query: {query}
    """
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        category_raw = completion.choices[0].message.content.strip()
        
        valid_categories = ["Coding", "Writing", "Summarisation", "Translation", "General reasoning"]
        for valid in valid_categories:
            if valid.lower() in category_raw.lower():
                return valid
        return "General reasoning"
    except Exception as e:
        st.error(f"Classification Error: {e}")
        return "General reasoning"

def generate_response(query: str, category: str):
    if category == "Coding":
        provider = "Groq (Llama-3.3 70B Versatile)"
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": query}],
            temperature=0.7,
        )
        response = completion.choices[0].message.content
    else:
        provider = "Gemini (1.5 Flash)"
        res = gemini_model.generate_content(query)
        response = res.text
        
    return provider, response

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            st.caption(message["metadata"])

# Chat Input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Classifying and routing..."):
            category = classify_intent(prompt)
            provider, response_text = generate_response(prompt, category)
            
            metadata = f"*(Routed to {provider} via **{category}**)*"
            
            st.markdown(response_text)
            st.caption(metadata)
            
    # Save to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "metadata": metadata
    })