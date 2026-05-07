import streamlit as st
import PyPDF2
import requests

GATEWAY_ENDPOINTS = {
    "US (default)": "https://llm-gateway.assemblyai.com/v1/chat/completions",
    "EU": "https://llm-gateway.eu.assemblyai.com/v1/chat/completions",
}

MODELS_BY_PROVIDER = {
    "Anthropic Claude": [
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
    ],
    "OpenAI GPT": [
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-oss-120b",
        "gpt-oss-20b",
        "gpt-5.5",
    ],
    "Google Gemini": [
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.1-flash-lite-preview",
    ],
    "Alibaba Cloud Qwen": [
        "qwen3-next-80b-a3b",
        "qwen3-32B",
    ],
    "Moonshot AI Kimi": [
        "kimi-k2.5",
    ],
}

US_ONLY_MODELS = {
    *MODELS_BY_PROVIDER["OpenAI GPT"],
    "gemini-3.1-flash-lite-preview",
}


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def process_with_llm_gateway(text, prompt, api_key, model, endpoint, max_tokens, temperature, system_prompt):
    try:
        headers = {
            "authorization": api_key,
            "content-type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": f"{prompt}\n\n---\nDocument:\n{text}",
        })

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = requests.post(endpoint, headers=headers, json=payload, timeout=180)

        if response.status_code == 200:
            result = response.json()
            choice = result["choices"][0]["message"]
            usage = result.get("usage", {})
            return {
                "content": choice.get("content", ""),
                "usage": usage,
                "model": result.get("model", model),
            }
        return {"error": f"{response.status_code} - {response.text}"}

    except Exception as e:
        return {"error": f"Error processing with LLM Gateway: {e}"}


st.set_page_config(page_title="PDF Analysis with LLM Gateway", layout="wide")

st.title("PDF Text Analysis with AssemblyAI LLM Gateway")
st.markdown(
    "Upload a PDF and analyze it with any of 25+ models from Anthropic, OpenAI, "
    "Google, Alibaba and Moonshot — all through AssemblyAI's unified LLM Gateway."
)

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

with st.sidebar:
    st.markdown("### Configuration")

    api_key = st.text_input(
        "AssemblyAI API key",
        value=st.session_state.api_key,
        type="password",
        help="Get your API key from https://www.assemblyai.com/dashboard/",
    )
    if api_key:
        st.session_state.api_key = api_key

    region = st.radio(
        "Region",
        options=list(GATEWAY_ENDPOINTS.keys()),
        index=0,
        help="EU keeps data in the European Union. OpenAI models and a few previews are US-only.",
    )

    provider = st.selectbox("Provider", options=list(MODELS_BY_PROVIDER.keys()), index=0)

    available_models = MODELS_BY_PROVIDER[provider]
    if region == "EU":
        available_models = [m for m in available_models if m not in US_ONLY_MODELS]

    if not available_models:
        st.warning(f"No {provider} models available in the EU region. Switch to US.")
        st.stop()

    selected_model = st.selectbox("Model", options=available_models, index=0)

    with st.expander("Advanced parameters"):
        max_tokens = st.slider("max_tokens", 256, 8192, 2000, step=128)
        temperature = st.slider("temperature", 0.0, 2.0, 0.7, step=0.1)
        system_prompt = st.text_area(
            "System prompt (optional)",
            value="You are a helpful assistant analyzing PDF documents. Be concise and accurate.",
            height=100,
        )

if not st.session_state.api_key:
    st.warning("Enter your AssemblyAI API key in the sidebar to continue.")
    st.stop()

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    text_content = extract_text_from_pdf(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Extracted Text")
        st.caption(f"{len(text_content):,} characters")
        st.text_area("PDF Content", text_content, height=400)

    with col2:
        st.markdown("### LLM Gateway Analysis")
        user_prompt = st.text_area(
            "Prompt",
            placeholder="e.g. Summarize the main points in bullet points",
            height=120,
        )

        if st.button(f"Analyze with {selected_model}") and user_prompt:
            with st.spinner(f"Calling {selected_model} via {region}..."):
                result = process_with_llm_gateway(
                    text_content,
                    user_prompt,
                    st.session_state.api_key,
                    selected_model,
                    GATEWAY_ENDPOINTS[region],
                    max_tokens,
                    temperature,
                    system_prompt,
                )

            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown("#### Results")
                st.write(result["content"])

                usage = result.get("usage") or {}
                if usage:
                    cols = st.columns(3)
                    cols[0].metric("Prompt tokens", usage.get("prompt_tokens", "—"))
                    cols[1].metric("Completion tokens", usage.get("completion_tokens", "—"))
                    cols[2].metric("Total tokens", usage.get("total_tokens", "—"))
                st.caption(f"Model: {result.get('model', selected_model)}")
