from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA

import gradio as gr

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

def get_llm():
    model_id = "qwen2.5:3b"

    llm =  OllamaLLM(model=model_id, temperature=0.7, max_tokens=2048)
    return llm

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

def embeddings():
    embedding = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    return embedding

def vector_database(chunks):
    embedding_model = embeddings()
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
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Here is your answer"),
    title="RAG QA BOT",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

rag_application.launch(server_name="127.0.0.1", server_port=7860, share=True)