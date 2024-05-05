import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.llms import Ollama
from langchain.embeddings import  HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplets import css, bot_template, user_template
from langchain.llms import HuggingFaceHub



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200, length_function = len)
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorestores = FAISS.from_texts(text = text_chunks, embeddings = embeddings)
    return vectorestores


def get_conversation(vectorstore):
    #llm = Ollama()
    llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length" : 512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_massages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                               retriver=vectorstore.as_retriever(),
                                                               memory=memory)
    return conversation_chain

def handel_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in  st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your PDFs :books")
    user_question = st.text_input("As a question about your documents:")
    if user_question:
        handel_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello Nutan"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get the PDF text
                raw_text = get_pdf_text(pdf_docs)
                


                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                # create vectore store
                vectorestore = get_vectorstore(text_chunks)

                # conversation chain
                st.session_state.conversation = get_conversation(vectorestore)

    st.session_state.conversation 


if __name__ == "__main__":
    main()