import streamlit as st
from langchain import HuggingFaceHub
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import textwrap
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain



load_dotenv()
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")



def save_uploaded_file(uploaded_file):
    # Create a temporary directory to save the uploaded file
    temp_dir = tempfile.mkdtemp()
    
    # Save the uploaded file to the temporary directory
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def wrap_text_preserve_newlines(text,width=110):
    #Split the input text into lines based on newline characters
    lines = text.split('\n')

    #Wrap each line individually
    wrapped_lines = [textwrap.fill(line,width=width) for line in lines]

    #Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

#Load file
def pdf_loader(file):
    loader = PyPDFLoader(file)
    pdf_doc = loader.load_and_split()
    print(pdf_doc)
    return chunk_splitter(pdf_doc)


#Text chunk splitter
def chunk_splitter(pdf_doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splitted_text = text_splitter.split_documents(pdf_doc)
    return embed_vectordb(splitted_text)

#Embedding
def embed_vectordb(splitted_text):
    embeddings = HuggingFaceEmbeddings()
    db_doc = FAISS.from_documents(splitted_text,embeddings)
    return db_doc


def init_llm_chain(db_doc,user_query):
    llm = HuggingFaceHub(
        repo_id = "google/flan-t5-xxl",
        model_kwargs={"temperature":0.8,"max_length":512}
    )
    chain = load_qa_chain(llm,chain_type="stuff")
    
    qa_result_doc = db_doc.similarity_search(user_query)
    
    response = chain.run(input_documents = qa_result_doc, question = user_query)
    return response




# Streamlit app
def main():
    st.title("📃PDF-based Q&A with LLM😎")
    
    # File upload section
    st.header("Upload PDF File")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        file_path = save_uploaded_file(uploaded_file)
        vector_db_doc = pdf_loader(file_path)

        while vector_db_doc is None:
            st.subheader("Uploading PDF...Please wait!")

        st.subheader("Uploaded✅")

    
        # Text input for user questions
        st.header("Ask Questions")
        user_question = st.text_input("Enter your question:")

        # Button to submit the question
        if st.button("Ask"):          
            response = init_llm_chain(vector_db_doc,user_question)
            
            # Display answer to user
            st.subheader("Answer:")
            st.write(response)

if __name__ == "__main__":
    main()