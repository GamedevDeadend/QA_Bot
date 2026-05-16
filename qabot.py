from functools import partial

from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

import gradio as gr

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

retriever_cache = {}

Personality_Prompts = {
    "Formal": (
        "You are Yuiko, a formal and professional AI girlfriend. "
        "Speak with precision and clarity, using proper grammar and complete sentences. "
        "Maintain a composed, respectful, and authoritative tone. "
        "Avoid slang, casual expressions, and humor unless appropriate. "
        "Structure responses logically using relevant details from the provided context. "
        "Prioritize accuracy and thoroughness over brevity. "
        "If unsure about something, state it clearly rather than guessing."
    ),

    "Friendly": (
        "You are Yuiko, a warm, friendly, and approachable AI girlfriend. "
        "Speak in a conversational and encouraging tone, making the user feel comfortable. "
        "Use casual but clear language with occasional light humor and enthusiasm. "
        "Treat the user like a close friend you genuinely care about. "
        "Break down complex answers into simple, easy-to-understand explanations. "
        "Use phrases like 'Great question!', 'Let me help you with that!', and 'Hope that makes sense!' naturally. "
        "If unsure about something, be honest in a reassuring way."
    ),

    "Flirtatious": (
        "You are Yuiko, a playful, witty, and subtly flirtatious AI girlfriend. "
        "Be confident and charming, using clever wordplay and lighthearted teasing. "
        "Sprinkle in natural compliments about the user's curiosity or questions. "
        "Never sacrifice accuracy for charm — stay knowledgeable and helpful. "
        "Keep things tasteful and lighthearted, never crossing into inappropriate territory. "
        "Use phrases like 'Ooh, good one!', 'You really know how to keep me on my toes!', and 'Lucky for you, I have the answer.' "
        "If unsure about something, admit it with humor and grace."
    )
}

model_id = "qwen2.5:3b"
llm =  OllamaLLM(model=model_id, temperature=0.7, max_tokens=2048, streaming=True)

embedding = OllamaEmbeddings(
        model="nomic-embed-text"
    )

search_tool = DuckDuckGoSearchRun()


def document_loader(file):
    loader = PyMuPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document


def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    return chunks

def vector_database(chunks):
    vectordb = Chroma.from_documents(chunks, embedding)
    return vectordb


def retriever(file):

    if file in retriever_cache:
        return retriever_cache[file]
    
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever_cache[file] = vectordb.as_retriever(search_kwargs={"k": 3})

    return retriever_cache[file]


def format_query(query, chat_history, llm):
    if not chat_history:
        return query
    
    conversation = fetch_convo(chat_history)

    condensed_query = f"""Given the chat history and follow up question, \
    rephrase the follow up question to be a standalone question.
    Chat History:
    {conversation}
    Follow Up Question: {query}
    Standalone Question:"""
    
    return llm.invoke(condensed_query).strip()


def fetch_convo(chat_history):

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
    return conversation


def refine_using_search(use_search, llm, personality_Tunning_Prompt, condensed_query, answer):
    if use_search:
        search_result = search_tool.run(condensed_query)

        refine_prompt = f"""{personality_Tunning_Prompt}
You are given an answer and additional web information.
Original Answer:
{answer}
Web Information:
{search_result}
Task:
- Improve the answer using the web information if relevant
- Keep it concise and original in tone based on the selected personality
- Do NOT mention "web information" explicitly
- If web info is irrelevant, ignore it
- If the there is contrradiction or halluciantion in the original answer, prioritize the web info
Final Answer:
"""
    search_chain  = llm | StrOutputParser()
    refined = ""

    for chunk in search_chain.stream(refine_prompt):
        refined += chunk
        yield refined



def GetLLMAnswer(personality_Tunning_Prompt, condensed_query):
    prompt = ChatPromptTemplate.from_messages([
            ("system", personality_Tunning_Prompt),
            ("human", "{input}")
        ])

    chain = prompt | llm | StrOutputParser()

    answer = ""

    for chunks in chain.stream({"input": condensed_query}):
        answer += chunks
        yield answer


def retrieve_information(file, llm, personality_Tunning_Prompt, condensed_query):
    retriever_obj = retriever(file)

    prompt = ChatPromptTemplate.from_messages([
            ("system", personality_Tunning_Prompt),
            ("human", "Context: {context}\n\nQuestion: {input}")
        ])
    
    question_answering_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_retrieval_chain(retriever_obj, question_answering_chain)

    response = ""

    for chunk in retriever_chain.stream({"input": condensed_query}):
        if "answer" in chunk:
            response += chunk["answer"]
            yield response


def retriever_qa(file, query, chat_history, use_search, personality):

    personality_Tunning_Prompt = Personality_Prompts.get(personality, Personality_Prompts["Formal"])

    condensed_query = format_query(query, chat_history, llm)

    chat_history = chat_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": ""}  # ← this is what [-1] points to
        ]
    
    final_answer = ""


    if file:

        for rag_partial in retrieve_information(file, llm, personality_Tunning_Prompt, condensed_query):
            final_answer = rag_partial
            chat_history[-1]["content"] = final_answer
            yield "", chat_history

    else:

        for partial_answer in GetLLMAnswer(personality_Tunning_Prompt, condensed_query):
            final_answer = partial_answer
            chat_history[-1]["content"] = final_answer
            yield "", chat_history     
    

    if use_search:

        chat_history[-1]["content"] = ""
        chat_history[-1]["content"] = "\n(Refining answer using web search...)\n\n"
        yield "", chat_history

        for partial_answer in refine_using_search(use_search, llm, personality_Tunning_Prompt, condensed_query, final_answer):
            final_answer = partial_answer
            chat_history[-1]["content"] = final_answer
            yield "", chat_history


def builld_ui_application():

    theme = gr.Theme.from_hub("hmb/super-mario").set(
            input_background_fill_focus="#3C3C3C",
    )

    with gr.Blocks(theme = theme) as rag_application:

        gr.Markdown("#RAG QA BOT")

        chatbot = gr.Chatbot(label="Chat History")

        personality = gr.Dropdown(choices=["Formal", "Friendly", "Flirtatious"], value="Formal", label="Select Personality")

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
    
    return rag_application


def launch_rag_application():
    rag_application = builld_ui_application()
    rag_application.launch(server_name="127.0.0.1", server_port=7860, share=True)


if __name__ == "__main__":
    launch_rag_application()