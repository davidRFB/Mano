"""
Mano LSC Translator - Streamlit App

This app provides a simple interface to capture images and translate them.
For the smoothest real-time experience, use the HTML frontend (index.html) instead.

Run with: streamlit run src/frontend/app.py
"""
import streamlit as st
import requests
from PIL import Image
import io

# --- Configuration ---
st.set_page_config(page_title="Mano LSC Translator", page_icon="🤟", layout="wide")

st.title("🤟 Mano: LSC Translator")

# Notice about HTML frontend
st.info("""
    **💡 For smooth real-time video** (like the desktop app), use the HTML frontend:
    1. Start the API: `uvicorn src.api.main:app --reload`
    2. Open `src/frontend/index.html` in your browser
    
    This Streamlit version uses camera snapshots instead.
""")

# 1. SIDEBAR: Connection Settings
st.sidebar.header("⚙️ Settings")
backend_option = st.sidebar.radio(
    "Select Backend:",
    ("Localhost (Docker)", "Cloud Run (Europe)")
)

CLOUD_URL = "https://mano-api-xyz123-ew.a.run.app" 
LOCAL_URL = "http://localhost:8000"

if backend_option == "Cloud Run (Europe)":
    API_URL = st.sidebar.text_input("API URL", value=CLOUD_URL)
else:
    API_URL = st.sidebar.text_input("API URL", value=LOCAL_URL)

# Stability settings
st.sidebar.header("🎯 Capture Settings")
STABILITY_THRESHOLD = st.sidebar.slider(
    "Same letter count to auto-add", 
    min_value=2, max_value=5, value=2
)

# --- SESSION STATE ---
if 'word_buffer' not in st.session_state:
    st.session_state.word_buffer = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_confidence' not in st.session_state:
    st.session_state.last_confidence = 0.0
if 'stability_counter' not in st.session_state:
    st.session_state.stability_counter = 0
if 'last_stable_letter' not in st.session_state:
    st.session_state.last_stable_letter = None
if 'capture_count' not in st.session_state:
    st.session_state.capture_count = 0


# --- MAIN INTERFACE ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Camera Capture")
    
    # Streamlit's native camera input
    img_file_buffer = st.camera_input(
        "Show a hand gesture and click capture",
        key=f"camera_{st.session_state.capture_count}"
    )

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        
        with st.spinner(f"Predicting..."):
            try:
                files = {"file": ("image.jpg", bytes_data, "image/jpeg")}
                response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    letter = result['letter']
                    confidence = result['confidence']
                    
                    st.session_state.last_prediction = letter
                    st.session_state.last_confidence = confidence
                    
                    # Display prediction with large letter
                    pred_col1, pred_col2 = st.columns([1, 2])
                    with pred_col1:
                        color = "green" if confidence > 0.8 else "orange"
                        st.markdown(
                            f"<h1 style='font-size: 6rem; color: {color}; "
                            f"text-align: center;'>{letter.upper()}</h1>",
                            unsafe_allow_html=True
                        )
                    with pred_col2:
                        st.metric("Confidence", f"{confidence:.1%}")
                        
                        # Stability tracking
                        if letter == st.session_state.last_stable_letter:
                            st.session_state.stability_counter += 1
                        else:
                            st.session_state.stability_counter = 1
                            st.session_state.last_stable_letter = letter
                        
                        st.progress(
                            min(st.session_state.stability_counter / STABILITY_THRESHOLD, 1.0),
                            text=f"Stability: {st.session_state.stability_counter}/{STABILITY_THRESHOLD}"
                        )
                        
                        # Auto-add when stable
                        if st.session_state.stability_counter >= STABILITY_THRESHOLD:
                            st.session_state.word_buffer.append(letter)
                            st.session_state.stability_counter = 0
                            st.session_state.last_stable_letter = None
                            st.success(f"✓ Added '{letter.upper()}' automatically!")
                            st.rerun()
                    
                    # Manual add button
                    if st.button(f"➕ Add '{letter.upper()}' to word", use_container_width=True):
                        st.session_state.word_buffer.append(letter)
                        st.session_state.stability_counter = 0
                        st.session_state.capture_count += 1
                        st.rerun()
                        
                else:
                    st.error(f"Server Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Connection Failed. Is the API running at {API_URL}?")
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    st.subheader("📝 Word Builder")
    
    current_word = "".join(st.session_state.word_buffer)
    
    if current_word:
        st.markdown(
            f"<h1 style='font-size: 3rem; color: #ffcc00; letter-spacing: 4px;'>"
            f"{current_word.upper()}</h1>",
            unsafe_allow_html=True
        )
    else:
        st.info("No letters yet. Capture gestures!")
    
    # Letter chips
    if st.session_state.word_buffer:
        st.markdown(
            " ".join([
                f"<span style='background: #333; padding: 5px 10px; "
                f"border-radius: 4px; margin: 2px;'>{l.upper()}</span>"
                for l in st.session_state.word_buffer
            ]),
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    # Controls
    col_a, col_b = st.columns(2)
    
    with col_a:
        if st.button("⬅️ Backspace", use_container_width=True):
            if st.session_state.word_buffer:
                st.session_state.word_buffer.pop()
                st.rerun()
    
    with col_b:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.word_buffer = []
            st.session_state.stability_counter = 0
            st.session_state.last_stable_letter = None
            st.rerun()
    
    st.markdown("---")
    
    if st.button("🤖 AI Correct", use_container_width=True, disabled=True):
        st.warning("LLM connection coming soon!")
    
    # Quick tips
    with st.expander("💡 Tips"):
        st.markdown("""
        - **Auto-capture**: Same letter detected multiple times gets added automatically
        - **For real-time video**: Use the HTML frontend (`index.html`)
        - **API not working?** Make sure to run `uvicorn src.api.main:app --reload`
        """)