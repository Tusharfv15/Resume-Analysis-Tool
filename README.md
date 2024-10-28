# 📝 CV Analysis Tool

A **Streamlit**-based application that analyzes PDF CVs by answering custom questions defined by the user. This tool leverages **OpenAI embeddings** and **LangChain** to extract and process relevant information from uploaded CVs, outputting answers that can be exported as a CSV file for further use.
![Screenshot 2024-10-28 155134](https://github.com/user-attachments/assets/9f542e08-946d-4ae6-9306-75fa45d8acc2)


## ✨ Features

- **📄 Upload and Analyze**: Upload a CV in PDF format and retrieve answers to predefined questions.
- **⚙️ Customizable Questions**: Easily add, edit, or remove questions to tailor the analysis.
- **🧠 AI-Powered Analysis**: Uses OpenAI embeddings through the LangChain framework to search and answer questions based on CV content.
- **📥 CSV Export**: Download the analysis results as a CSV file for easy sharing and record-keeping.
- **📊 Export Results**: Export analyzed data to a CSV file for easy sharing and record-keeping.


![Screenshot 2024-10-28 155149](https://github.com/user-attachments/assets/e5dc4a82-3562-446c-b9bc-a5051e9e88e2)
![image](https://github.com/user-attachments/assets/52bfdd46-3c37-4baa-8b7a-ed05f9b8935b)



## 💼 Key Technologies

- **Streamlit**: Web framework for building interactive web applications in Python.
- **LangChain**: Framework for managing language models and chains, utilized here to handle document embedding and question-answering processes.
- **OpenAI**: Provides embeddings used for CV data analysis.
- **FAISS**: For efficient similarity search, aiding in finding the most relevant sections in the CV.
- **PyPDF2**: PDF handling and text extraction library.
- **Dotenv**: Environment variable management.

## 📂 Project Structure

```plaintext
resume-analysis-tool/
├── app.py                # Main Streamlit application file
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables
├── README.md             # Project documentation
