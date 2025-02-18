import streamlit as st 
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
# Title and subtitle
st.title("YouTube Video Summary Generator")
st.subheader("Enter a YouTube URL and get a summary of the video.")

# Sidebar for user inputs
st.sidebar.header("Settings")
url = st.sidebar.text_input("Enter YouTube URL", "")
summarize_button = st.sidebar.button("Generate Summary")

# If the user clicks the button, summarize the video
if summarize_button and url:
    with st.spinner("Fetching video transcript..."):
        loader = YoutubeLoader.from_youtube_url(youtube_url=url, language=["en", "en-US"])
        transcript = loader.load()  # Fetch video transcript
        st.write("**Transcript fetched successfully!**")

        # Split the transcript into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(transcript)
        st.write(f"Transcript divided into {len(chunks)} chunks.")

        # Set up the summarization model
        api_key = os.getenv("GROQ_API_KEY")
        model = ChatGroq(model="llama3-8b-8192", temperature=0.7,api_key=api_key)

        summarize_chain = load_summarize_chain(llm=model, chain_type="refine", verbose=True)
        
        with st.spinner("Generating summary..."):
            summary = summarize_chain.run(chunks)
        
        # Display the generated summary
        st.write("**Summary of the Video:**")
        st.write(summary)

# Styling
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background-color: #f1f3f4;
        }
        .stButton>button {
            background-color: #6200EE;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #3700B3;
        }
    </style>
""", unsafe_allow_html=True)
