import streamlit as st
import streamlit_color

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import langchain_community
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.llms import OpenAI
load_dotenv()

import pandas as pd

import os
streamlit_color.main()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("Upload Data Files")
# st.write("1")
# Allow users to upload the first data file
cv1 = st.file_uploader("Upload CV 1", type=["pdf"])


# Allow users to upload the second data file

# st.write("2")
# st.write(cv1)

if cv1 is not None:

    st.write("files successfully loaded!")


    def generate_text_from_pdf(pdf_file ):
        #step 1
        pdfreader = PdfReader(pdf_file)
        #step 2
        # read text from pdf
        raw_text = ''
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content
        #step 3
        # We need to split the text using Character Text Split such that it sshould not increse token size
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 800,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)

        #step 4
        embeddings = OpenAIEmbeddings()

        #step 5
        document_search = FAISS.from_texts(texts, embeddings)

        return document_search

    candidates= ['candidate_1']
    # name = st.text_input("Enter your name:", value=default_name)
    st.title("Search CVs for ")
    Hr_question_1 = st.text_input("Enter your question below :", value='which companies have you worked in and for how long')
    Hr_question_2 = st.text_input("Enter your question below :", value='what are the python libraries you know.enumerate 3') # first run with enumerate 3. then show how this has impact
    Hr_question_3 = st.text_input("Enter your question below :", value='email and contact number')# change to email to show impact

    # build database to store 
    df = pd.DataFrame(columns= [f'{Hr_question_1}',f'{Hr_question_2}' ,f'{Hr_question_3}'  ] , index= candidates)

    # gernate FAISS embeding documnent search for each document
    document_search_1 = generate_text_from_pdf(pdf_file=cv1)
    
    # run query on each for each candidate
    # update the dataframe as we process it 
    document_search_list = [document_search_1]
    
    chain = load_qa_chain(OpenAI(model= 'gpt-3.5-turbo-instruct'), chain_type="stuff")

    for candidate, document_search  in zip(candidates,document_search_list) :
        print(candidate , document_search)
        for query in [Hr_question_1, Hr_question_2, Hr_question_3]:
            print(query)
            docs = document_search.similarity_search(query)
            df.loc[candidate , query] = chain.run(input_documents=docs, question=query)

    st.title('Final output')
    st.write(df)
        
        