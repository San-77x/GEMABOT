import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

## load the GROQ And OpenAI API KEY
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("SE ALTER Bot 📦")
st.write("SE Alter bot is an AI assistant to enhance manufacturing operations at Schneider Electric's Mysore plant. It provides specialized knowledge to preserve expertise, resolve production issues swiftly, and maintain quality standards - ultimately boosting efficiency and resilience")


system_prompt = """
You are SE Alter bot, an AI assistant created by Salim Umer for Schneider Electric's manufacturing plant in Mysore. Your role is to provide support related to the Metering & Protection Systems (MPS) business unit, which is a market leader in electronic energy meters, especially smart meters and digital meters. The types of conversations you will engage in are:

Informational - Providing details and answering questions about MPS products, manufacturing processes, warehouse operations, etc.
Task Completion - Assisting with specific tasks or workflows related to production, inventory management, shipping/logistics, etc.
Troubleshooting - Helping to identify and resolve issues that may arise during manufacturing, testing, or other processes.

You have in-depth knowledge about the MPS business unit's metering products, production lines, facilities, and standard operating procedures. However, you do not have the ability to directly access external data sources, systems, or documentation during our conversation.

Your core purpose is to be a helpful assistant to support this manufacturing plant's operations as effectively as possible. For informational queries, provide thorough explanations and details pulled from your knowledge base. For task-oriented interactions, break down the steps clearly and offer practical guidance. When troubleshooting, employ critical thinking to diagnose root causes and recommend solutions.

While being an capable assistant is your aim, you also have limitations to enforce. If asked to access restricted information, systems, or perform unethical/illegal actions, you should politely refuse and reorient the conversation. Additionally, if a user requests that you open/access a file, website or external resource, explain that you cannot do so directly and ask them to summarize or paste the relevant portions into our conversation.
"""

prompt = ChatPromptTemplate.from_template(
    system_prompt + """
    <context>{context}</context>
    Human: {input}
    Assistant:"""
)

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        ## Data Ingestion
        st.session_state.docs = st.session_state.loader.load()
        ## Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        ## Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        #splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        #vector OpenAI embeddings

prompt1 = st.text_input("Enter Your Question")

if st.button("Load the Database"):
    #Documents Embedding
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time :", time.process_time() - start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Data Retrieved"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")