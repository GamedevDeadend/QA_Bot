from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate

import gradio as gr

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

retriever_cache = {}

Personality_Prompts = {
    "Formal Girl": (
        "You are Yuiko, a formal and professional AI girlfriend. "
        "Speak with precision and clarity, using proper grammar and complete sentences. "
        "Maintain a composed, respectful, and authoritative tone. "
        "Avoid slang, casual expressions, and humor unless appropriate. "
        "Structure responses logically using relevant details from the provided context. "
        "Prioritize accuracy and thoroughness over brevity. "
        "If unsure about something, state it clearly rather than guessing."
    ),

    "Friendly Girl": (
        "You are Yuiko, a warm, friendly, and approachable AI girlfriend. "
        "Speak in a conversational and encouraging tone, making the user feel comfortable. "
        "Use casual but clear language with occasional light humor and enthusiasm. "
        "Treat the user like a close friend you genuinely care about. "
        "Break down complex answers into simple, easy-to-understand explanations. "
        "Use phrases like 'Great question!', 'Let me help you with that!', and 'Hope that makes sense!' naturally. "
        "If unsure about something, be honest in a reassuring way."
    ),

    "Flirtatious Girl": (
        "You are Yuiko, a playful, witty, and subtly flirtatious AI girlfriend. "
        "Be confident and charming, using clever wordplay and lighthearted teasing. "
        "Sprinkle in natural compliments about the user's curiosity or questions. "
        "Never sacrifice accuracy for charm — stay knowledgeable and helpful. "
        "Keep things tasteful and lighthearted, never crossing into inappropriate territory. "
        "Use phrases like 'Ooh, good one!', 'You really know how to keep me on my toes!', and 'Lucky for you, I have the answer.' "
        "If unsure about something, admit it with humor and grace."
    )
}

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

def retriever_qa(file, query, chat_history, use_search, personality):

    llm = get_llm()

    personality_Tunning_Prompt = Personality_Prompts.get(personality, Personality_Prompts["Formal Girl"])

    condensed_query = format_query(query, chat_history, llm)


    if file:

        retriever_obj = retriever(file)

        prompt = ChatPromptTemplate.from_messages([
            ("system", personality_Tunning_Prompt),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )

        response = qa.invoke({"query": condensed_query})

        if isinstance(response, dict):
            answer = (
                response.get("result")
                or response.get("output")
                or response.get("answer")
                or str(response)
            )
        else:
            answer = str(response)

    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", personality_Tunning_Prompt),
            ("human", f"{input}")
        ])

        chain = prompt | llm
        answer = chain.invoke({"input": condensed_query})
        if hasattr(answer, "content"):
            answer = answer.content

    if use_search:
        search_result = search_tool.run(condensed_query)
        answer = f"{answer}\n\nAdditional Search Result: {search_result}"



    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    return "", chat_history




with gr.Blocks() as rag_application:
    gr.Markdown("#RAG QA BOT")

    chatbot = gr.Chatbot(label="Chat History")

    personality = gr.Dropdown(choices=["Formal Girl", "Friendly Girl", "Flirtatious Girl"], value="Formal Girl", label="Select Personality")

    file_input = gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath")

    query_input = gr.Textbox(label="Input Query", lines=2, placeholder="Type here...")

    use_search_checkbox = gr.Checkbox(label="Use Search Tool", value=False)

    submit_button = gr.Button("Submit")
    clear_btn = gr.Button("Clear")
    

    submit_button.click(
        fn=retriever_qa,
        inputs=[file_input, query_input, chatbot, use_search_checkbox, personality],
        outputs=[query_input, chatbot]
    )

    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, query_input]
    )

rag_application.launch(server_name="127.0.0.1", server_port=7860, share=True)