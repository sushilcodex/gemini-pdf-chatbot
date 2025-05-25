from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from PIL import Image
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    Context:\n {context} \n
    Question:\n {question} \n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    import os
    if not os.path.exists("faiss_index"):
        st.error("Please upload and process a PDF document first!")
        return
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversation_chain(new_db)
    
    response = chain(
        {
            "input_documents": docs,
            "question": user_question,
             
        },
        return_only_outputs=True
    )
    print(response)
    st.write("reply: ", response["output_text"])

def main():
    st.set_page_config(
        page_title="Gemini PDF Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.header("Gemini PDF Chat")
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
        user_input(user_question)   
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")
                # get the text chunks

if __name__ == '__main__':
    main()