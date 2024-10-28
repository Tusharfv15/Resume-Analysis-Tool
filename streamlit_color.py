import streamlit as st
def main():
    st.set_page_config(
        # page_title="Modern Color Palette",
        page_icon=":art:",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Modern color palette
    # primary_color = "#3498db"  # Blue
    # secondary_color = "#2ecc71"  # Green
    # background_color = "#f0f2f6"  # Light gray
    # text_color = "#2c3e50"  # Dark gray
    
    primary_color="#ff4b4b",
    secondary_color="#ffffff",
    background_color="#f0f2f6",
    text_color="#31333F"

    # CSS styles
    custom_css = f"""
        .stApp {{
            background-color: {background_color};
            color: {text_color};
        }}
        .sidebar .sidebar-content {{
            background-color: {secondary_color};
            color: white;
        }}
        .css-17eq0hr {{
            color: {primary_color};
        }}
        
    """
    st.write("<style>{}</style>".format(custom_css), unsafe_allow_html=True)

    # st.title("Modern Color Palette")
    # st.markdown("This is a Streamlit app with a modern color palette.")
