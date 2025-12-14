import streamlit as st
import requests
from PIL import Image
import io
import time

# --- Configuration ---
st.set_page_config(page_title="Mano LSC Translator", page_icon="🤟")

st.title("🤟 Mano: LSC Translator")
st.markdown("Take a photo of a hand gesture (A-Z) to translate it.")

# 1. SIDEBAR: Connection Settings
st.sidebar.header("⚙️ Settings")
backend_option = st.sidebar.radio(
    "Select Backend:",
    ("Cloud Run (Europe)", "Localhost (Docker)")
)

# Paste your actual Cloud Run URL here (the one you got from gcloud)
CLOUD_URL = "https://mano-api-xyz123-ew.a.run.app" 
LOCAL_URL = "http://localhost:8000"

if backend_option == "Cloud Run (Europe)":
    API_URL = st.sidebar.text_input("API URL", value=CLOUD_URL)
else:
    API_URL = st.sidebar.text_input("API URL", value=LOCAL_URL)

# --- SESSION STATE: To store the word being built ---
if 'word_buffer' not in st.session_state:
    st.session_state.word_buffer = []

# --- MAIN INTERFACE ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Camera Feed")
    # Streamlit's native camera input (Simpler/More stable than WebRTC for v1)
    img_file_buffer = st.camera_input("Capture Gesture")

    if img_file_buffer is not None:
        # Convert the buffer to bytes
        bytes_data = img_file_buffer.getvalue()
        
        # Show a spinner while waiting for the API
        with st.spinner(f"Sending to {backend_option}..."):
            try:
                # Prepare the file for upload
                files = {"file": ("image.jpg", bytes_data, "image/jpeg")}
                
                # Send POST request
                response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    letter = result['letter']
                    confidence = result['confidence']
                    
                    # Display Result
                    st.success(f"**Prediction:** {letter}")
                    st.caption(f"Confidence: {confidence:.2%}")
                    
                    # Add to Word Button
                    if st.button(f"Add '{letter}' to Word"):
                        st.session_state.word_buffer.append(letter)
                        st.rerun() # Refresh to update the list
                else:
                    st.error(f"Server Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Connection Failed. Is the server running/URL correct?")
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    st.subheader("📝 Word Builder")
    
    # Display the current word
    current_word = "".join(st.session_state.word_buffer)
    
    if current_word:
        st.markdown(f"# {current_word}")
    else:
        st.info("No letters yet.")

    st.markdown("---")
    
    # Controls
    if st.button("⬅️ Backspace"):
        if st.session_state.word_buffer:
            st.session_state.word_buffer.pop()
            st.rerun()

    if st.button("🗑️ Clear All"):
        st.session_state.word_buffer = []
        st.rerun()

    if st.button("🤖 AI Correct (Coming Soon)"):
        st.warning("LLM connection not set up yet!")