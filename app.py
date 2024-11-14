import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub
import regex as re
import tempfile

# Load API key securely
LLM_API_KEY = st.secrets["LLM_API_KEYS"]

def generate_response(file_content, query_text):
    # Load document from file content instead of file name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    # Load and split PDF document
    pdfloader = PyPDFLoader(temp_file_path)
    documents = pdfloader.load_and_split()

    # Initialize embeddings using Hugging Face API
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=LLM_API_KEY, model_name="thenlper/gte-large")

    # Create a vectorstore with Chroma in memory mode (avoids SQLite)
    db = Chroma.from_documents(documents, embeddings, persist_directory=None)

    # Create a retriever
    retriever = db.as_retriever()

    # Initialize the Hugging Face model for QA
    model = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=LLM_API_KEY,
        model_kwargs={"temperature": 0.01, "max_new_tokens": 1024}
    )

    # Create a QA chain
    qa = RetrievalQA.from_chain_type(llm=model, chain_type='stuff', retriever=retriever)

    # Run the QA model on the query
    return qa.run(query_text)

# Streamlit UI
st.set_page_config(page_title='🦜🔗 Document Chat Powered by Huggingface')
st.title('🦜🔗 Ask the Document !!!!')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

# Form submission
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not (uploaded_file and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            # Read file content and pass it directly
            file_content = uploaded_file.getvalue()
            response = generate_response(file_content, query_text)
            result.append(response)

            # Extract helpful answer
            match = re.search(r'Helpful Answer: (.+)', response)
            if match:
                helpful_answer = match.group(1).strip()
            else:
                helpful_answer = response  # Fallback if regex doesn't match

# Display the result
if result:
    st.info(helpful_answer)
