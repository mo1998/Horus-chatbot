#importing essential libraries
import streamlit as st
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.vectorstores import Chroma
from langchain.utilities import GoogleSearchAPIWrapper
import torch
from dotenv import load_dotenv
import os

#load envrioment variables
load_dotenv()

#create api tokens
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
auth_token = os.getenv('HUGGING_TOKEN')

def create_llm():
    model_name = "meta-llama/Llama-2-7b-chat-hf"

    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir='./model/', use_auth_token=auth_token)
    #load model
    model = AutoModelForCausalLM.from_pretrained(model_name,cache_dir='./model/', use_auth_token=auth_token, torch_dtype=torch.float32,rope_scaling={"type": "dynamic", "factor": 2}, load_in_4bit=True)
    #create pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
        eos_token_id=tokenizer.eos_token_id
    )
    #create llm model
    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0.7})
    return llm

def create_embeddings():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    return embeddings


def create_webRetriever(embeddings,llm):
    #create vector store
    vectorstore = Chroma(embedding_function=embeddings,persist_directory="./chroma_db_oai")
    #create google search module
    search = GoogleSearchAPIWrapper()
    #web retriever
    web_retriever = WebResearchRetriever.from_llm(vectorstore=vectorstore,llm=llm, search=search, num_search_results=3)
    return web_retriever


#set configuration of web app
st.set_page_config(page_title="HORUS CHATBOT",page_icon= ':eagle:')


#main flow of web app
if 'retriever' not in st.session_state:
    embeddings = create_embeddings()
    llm = create_llm()
    st.session_state['retriever'] = create_webRetriever(embeddings=embeddings,llm=llm)
    st.session_state['llm'] = llm
web_retriever = st.session_state.retriever
llm = st.session_state.llm


q = st.chat_input("Tell me what do you want to know?")

if 'history' not in st.session_state:
    st.session_state['history'] = []

if q:
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)
    result = qa_chain({"question": q})
    r = result['answer']
    st.session_state.history.append([q,r]) 
    for i in range(len(st.session_state.history)):
        with st.chat_message("ai",avatar="🦅"):
            st.write(st.session_state.history[i][1])
        with st.chat_message("user"):
            st.write(st.session_state.history[i][0])
    #print(st.session_state.history)
