import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os 
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate



load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_pdf_text(pdf_docs):
    text= "" # Initialize an empty string to hold the text
    # Loop through each uploaded PDF document
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # Iterate through each page in the PDF
        for page in pdf_reader.pages:
            text += page.extract_text()  # Append the text of each page to the text variable
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt_template= """
    Answer the question as deatailed as possible from the provided context.
    make sure to provide all the relevant information.
    If you don't know the answer, just say that you don't know. DO NOT try to make up an answer.
    Context: \n{context}?\n
    Question: \n{question}?\n

    Answer:

"""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
 
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversation_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, 
        return_only_outputs=True
    )
    st.write("Your question: ", user_question)
    st.write("Reply: ", response['output_text'])


st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
st.header("Chat with Multiple PDFs :books:")
user_question = st.text_input("Ask a question about your documents:")
if user_question:
    user_input(user_question)

with st.sidebar:
    st.title("Menu: ")
    pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing"):
            raw_text= get_pdf_text(pdf_docs)
            text_chunks= get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Your PDFs have been processed successfully!")
            st.balloons()