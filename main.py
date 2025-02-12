import streamlit as st
import PyPDF2
import requests

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def process_with_lemur(text, prompt, api_key, model):
    try:
        headers = {
            "Authorization": api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "final_model": model,
            "prompt": prompt,
            "input_text": text,
            "max_output_size": 4000,
        }
        
        response = requests.post(
            "https://api.assemblyai.com/lemur/v3/generate/task",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error processing with LeMUR: {str(e)}"

# Set page configuration
st.set_page_config(page_title="PDF Analysis with LeMUR", layout="wide")

# Title and description
st.title("PDF Text Analysis with LeMUR")
st.markdown("""Upload a PDF file and analyze its content using AssemblyAI's LeMUR model. 
Start by entering your AssemblyAI API key below.""")

# Available models
LEMUR_MODELS = [
    "anthropic/claude-3-5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-sonnet",
    "assemblyai/mistral-7b"
]

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

# Sidebar configuration
with st.sidebar:
    st.markdown("### Configuration")
    
    # API key input
    api_key = st.text_input(
        "Enter your AssemblyAI API key",
        value=st.session_state.api_key,
        type="password",
        help="Get your API key from https://www.assemblyai.com/dashboard/"
    )
    
    if api_key:
        st.session_state.api_key = api_key
    
    # Model selection
    selected_model = st.selectbox(
        "Select LeMUR Model",
        options=LEMUR_MODELS,
        index=0,
        help="Choose the model to process your text"
    )

# Main content
if not st.session_state.api_key:
    st.warning("Please enter your AssemblyAI API key in the sidebar to continue.")
    st.stop()

# File upload and processing
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    # Process the PDF file
    text_content = extract_text_from_pdf(uploaded_file)
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Extracted Text")
        st.text_area(
            "PDF Content",
            text_content,
            height=400
        )
    
    with col2:
        st.markdown("### LeMUR Analysis")
        user_prompt = st.text_area(
            "Enter your prompt for analysis",
            placeholder="Example: Summarize the main points of this text in bullet points",
            height=100
        )
        
        if st.button("Analyze with LeMUR") and user_prompt:
            with st.spinner(f"Processing with LeMUR using {selected_model}..."):
                lemur_response = process_with_lemur(
                    text_content,
                    user_prompt,
                    st.session_state.api_key,
                    selected_model
                )
                st.markdown("#### Results:")
                st.write(lemur_response)
