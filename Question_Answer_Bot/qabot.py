from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import gradio as gr

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

def get_llm():
    model_id = "mistralai/mistral-small-3-1-24b-instruct-2503"
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=parameters,
    )
    return watsonx_llm

def document_loader(file):
    loader = PyPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document

def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    return chunks

def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        # Removed RETURN_OPTIONS dict — nested dicts cause unhashable type errors
    }
    embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return embedding

def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()

def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False
    )
    response = qa.invoke(query)
    if isinstance(response, dict):
        return (
            response.get("result")
            or response.get("output")
            or response.get("answer")
            or str(response)
        )
    return str(response)

rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Here is your answer"),
    title="RAG QA BOT",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

rag_application.launch(server_name="0.0.0.0", server_port=7860, share=True)