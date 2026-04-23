from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun

import gradio as gr

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

retriever_cache = {}
search_tool = DuckDuckGoSearchRun()

def get_llm():
    model_id = "qwen2.5:3b"

    llm =  OllamaLLM(model=model_id, temperature=0.7, max_tokens=2048)
    return llm

def document_loader(file):
    loader = PyMuPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document

def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=140,
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

    if file in retriever_cache:
        return retriever_cache[file]
    

    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever_cache[file] = vectordb.as_retriever()

    return retriever_cache[file]

def format_query(query, chat_history, llm):
    if not chat_history:
        return query

    conversation =  ""
    for message in chat_history:
        role = message.get("role", "")

        # content is a list of dicts with "text" key
        content = message.get("content", "")

        if isinstance(content, list):
            content = content[0].get("text", "")
        
        if role == "user":
            conversation += f"Human: {content}\n"
        elif role == "assistant":
            conversation += f"AI: {content}\n"

    condensed_query = f"""Given the chat history and follow up question, \
    rephrase the follow up question to be a standalone question.
    Chat History:
    {conversation}
    Follow Up Question: {query}
    Standalone Question:"""
    
    return llm.invoke(condensed_query).strip()

def retriever_qa(file, query, chat_history, use_search):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False
    )

    condensed_query = format_query(query, chat_history, llm)

    response = qa.invoke(condensed_query)

    if isinstance(response, dict):
        answer =  (
            response.get("result")
            or response.get("output")
            or response.get("answer")
            or str(response)
        )
    else:
        answer = str(response)

    if use_search:
        search_result = search_tool.run(condensed_query)
        answer = f"{answer}\n\nAdditional Search Result: {search_result}"



    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    return "", chat_history




with gr.Blocks() as rag_application:
    gr.Markdown("#RAG QA BOT")
    chatbot = gr.Chatbot(label="Chat History")
    file_input = gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath")
    query_input = gr.Textbox(label="Input Query", lines=2, placeholder="Type here...")
    use_search_checkbox = gr.Checkbox(label="Use Search Tool", value=False)
    submit_button = gr.Button("Submit")
    clear_btn = gr.Button("Clear")
    

    submit_button.click(
        fn=retriever_qa,
        inputs=[file_input, query_input, chatbot, use_search_checkbox],
        outputs=[query_input, chatbot]
    )

    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, query_input]
    )

rag_application.launch(server_name="127.0.0.1", server_port=7860, share=True)