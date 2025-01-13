import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Arxiv and Wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Function to extract PDF text
def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    full_text = ''
    for page in reader.pages:
        full_text += page.extract_text()
    return full_text

# Streamlit UI
st.set_page_config(page_title="Search Engine", layout="wide")

# Title and Sidebar Settings
st.title("Intelligent Search Engine")

# Sidebar for settings and API Key
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")
pdf_upload = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# Displaying previous messages in chat
if "messages" not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'Hi, I am a chatbot who can search the web. How can I help you?'}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Handling user input and integrating agents
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    st.chat_message('user').write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name='gemma2-9b-it', streaming=True)

    # Initialize list of tools
    tools = [search, arxiv, wiki]

    # If PDF is uploaded, add PDF query logic directly
    if pdf_upload:
        pdf_text = extract_pdf_text(pdf_upload)
        pdf_tool = DuckDuckGoSearchRun(name="PDF Tool", query=pdf_text)  # Treating PDF text as a query source
        tools.append(pdf_tool)

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_error=True)

    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)

# For better UI, added some sections and separators
st.markdown("---")
st.sidebar.markdown("### Agents Overview")
st.sidebar.markdown(
    """
    - **Groq**: AI-powered response generation.
    - **DuckDuckGo**: Web search engine.
    - **Arxiv**: Research paper search.
    - **Wikipedia**: Quick Wikipedia searches.
    - **PDF Agent**: Extracts and queries text from uploaded PDFs.
    """
)
