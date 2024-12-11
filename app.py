"""
Speaker Similarity Detection System - Streamlit Application
Author: Ken Maeda
"""

import streamlit as st
import torch
from typing import Dict, List
from tts_asr_eval_suite.secs import SECS


# Set page config
st.set_page_config(
    page_title="Speaker Similarity Detection",
    page_icon="🎤",
    layout="wide",
)

# Initialize session state
if 'similarity_results' not in st.session_state:
    st.session_state.similarity_results = None


def get_device() -> str:
    """Get the appropriate device (CPU/CUDA)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def process_audio_files(file1_path: str, file2_path: str, selected_methods: List[str]) -> Dict:
    """Process audio files and return similarity scores."""
    device = get_device()

    # Initialize SECS with selected methods
    secs = SECS(device=device, methods=selected_methods)

    # Calculate similarity
    try:
        similarity_scores = secs(file1_path, file2_path)
        return similarity_scores
    except Exception as e:
        st.error(f"Error processing audio files: {str(e)}")
        return None


def main():
    st.title("🎤 Speaker Similarity Detection System")

    # Sidebar for model selection
    st.sidebar.title("Model Settings")

    available_methods = [
        'resemblyzer',
        'wavlm_large_sv',
        'wavlm_base_plus_sv',
        'ecapa2'
    ]

    selected_methods = st.sidebar.multiselect(
        "Select Methods",
        available_methods,
        default=['ecapa2'],
        help="Choose one or more methods for similarity detection"
    )

    if not selected_methods:
        st.warning("Please select at least one method from the sidebar!")
        return

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("First Audio File")
        file1 = st.file_uploader("Upload first audio file", type=['wav', 'mp3'], key="file1")

    with col2:
        st.subheader("Second Audio File")
        file2 = st.file_uploader("Upload second audio file", type=['wav', 'mp3'], key="file2")

    # Process button
    if st.button("Compare Voices", disabled=not (file1 and file2)):
        if file1 and file2:
            with st.spinner("Processing audio files..."):
                # Process files
                similarity_scores = process_audio_files(file1, file2, selected_methods)

                if similarity_scores:
                    st.session_state.similarity_results = similarity_scores

    # Display results
    if st.session_state.similarity_results:
        st.subheader("Similarity Results")

        # Create columns for each score
        cols = st.columns(len(st.session_state.similarity_results))

        for col, (method, score) in zip(cols, st.session_state.similarity_results.items()):
            with col:
                st.metric(
                    label=method,
                    value=f"{score:.3f}",
                    help="Score range: -1 to 1 (higher means more similar)"
                )

        # Additional information
        st.info("""
        📊 Score Interpretation:
        - 1.0: Perfect similarity
        - 0.0: No similarity
        - -1.0: Perfect dissimilarity
        """)

    # Model Information
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        ### Available Models

        1. **ECAPA2**
        - Lightweight and efficient speaker verification model
        - Pretrained model available from 'Jenthe/ECAPA2'

        2. **Resemblyzer**
        - Real-time voice similarity embedding system
        - Uses pretrained VoiceEncoder model

        3. **WavLM Large SV**
        - Large-scale speaker verification model
        - Pretrained from 'subatomicseer/wavlm-large-sv-ckpts'

        4. **WavLM Base Plus SV**
        - Base model for speaker verification
        - Pretrained from 'microsoft/wavlm-base-plus-sv'
        """)


if __name__ == "__main__":
    main()
