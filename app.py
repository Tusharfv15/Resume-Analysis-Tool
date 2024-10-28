import streamlit as st
import streamlit_color
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.llms import OpenAI
import pandas as pd
import os

load_dotenv()
streamlit_color.main()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("CV Analysis Tool")

# Initialize questions in session state if not present
if 'questions' not in st.session_state:
    st.session_state.questions = [
        'Which companies have you worked in and for how long?',
        'What are the python libraries you know?',
        'What is your email and contact number?'
    ]

# File upload
cv1 = st.file_uploader("Upload CV", type=["pdf"])

# Question management section
st.subheader("Manage Questions")

# Add new question
new_question = st.text_input("Add a new question:")
if st.button("Add Question"):
    if new_question and new_question not in st.session_state.questions:
        st.session_state.questions.append(new_question)
        st.success(f"Added question: {new_question}")
    elif not new_question:
        st.warning("Please enter a question first")
    else:
        st.warning("This question already exists")

# Display and manage existing questions
st.subheader("Current Questions")
questions_to_remove = []

for i, question in enumerate(st.session_state.questions):
    col1, col2 = st.columns([4, 1])
    with col1:
        st.session_state.questions[i] = st.text_input(f"Question {i+1}", value=question, key=f"q_{i}")
    with col2:
        if st.button("Remove", key=f"remove_{i}"):
            questions_to_remove.append(i)

# Remove marked questions
for index in sorted(questions_to_remove, reverse=True):
    st.session_state.questions.pop(index)

def generate_text_from_pdf(pdf_file):
    pdfreader = PdfReader(pdf_file)
    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)
    return document_search

if cv1 is not None:
    st.success("CV successfully loaded!")
    
    # Create DataFrame with dynamic columns
    candidates = ['candidate_1']
    df = pd.DataFrame(columns=st.session_state.questions, index=candidates)
    
    # Process CV
    document_search = generate_text_from_pdf(pdf_file=cv1)
    chain = load_qa_chain(OpenAI(model='gpt-3.5-turbo-instruct'), chain_type="stuff")
    
    # Analysis progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process each question
    for i, query in enumerate(st.session_state.questions):
        status_text.text(f"Analyzing question {i+1} of {len(st.session_state.questions)}")
        docs = document_search.similarity_search(query)
        df.loc['candidate_1', query] = chain.run(input_documents=docs, question=query)
        progress_bar.progress((i + 1) / len(st.session_state.questions))
    
    status_text.text("Analysis complete!")
    
    # Display results
    st.title('Analysis Results')
    st.dataframe(df)
    
    # Export option
    if st.button("Export to CSV"):
        df.to_csv("cv_analysis_results.csv")
        st.success("Results exported to cv_analysis_results.csv")
