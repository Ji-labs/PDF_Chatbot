import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import torch

# Load the GPT-J model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

st.set_page_config(page_title="Chat with PDF", page_icon="📚")
st.title("Chat with your PDF 📚")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a conversational retrieval chain
def get_conversation_chain(vectorstore):
    embeddings = HuggingFaceEmbeddings()
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="You are an expert PDF assistant. Use the following context to answer questions accurately.\n"
                 "Provide clear, concise responses. If unsure, say so.\n\n"
                 "{context}\n"
                 "Question: {question}\n"
                 "Answer:"
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )
    return conversation_chain

# Sidebar for PDF upload
with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader("Upload your PDFs here", type="pdf", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing your PDFs..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            embeddings = HuggingFaceEmbeddings()
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.session_state.processComplete = True

# Main chat interface
if st.session_state.get("processComplete"):
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        response = st.session_state.conversation({"question": user_question})
        st.write(response["answer"])
else:
    st.write("👈 Upload PDFs to begin chatting!")
