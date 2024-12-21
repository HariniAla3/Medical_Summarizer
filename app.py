import streamlit as st
import whisper
import tempfile
import torch
import os
import requests
from datetime import datetime
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from datasets import load_dataset

# Replace with your Groq API key
API_KEY = "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"

# Set up headers for Groq API authentication
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'generated_summary' not in st.session_state:
    st.session_state.generated_summary = None
if 'current_summary' not in st.session_state:
    st.session_state.current_summary = None
if 'initial_summary_generated' not in st.session_state:
    st.session_state.initial_summary_generated = False
if 'medical_codes' not in st.session_state:
    st.session_state.medical_codes = None
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""

def set_page(page_name):
    st.session_state.page = page_name

# Load Whisper model for transcription
@st.cache_resource
def load_whisper_model():
    whisper_model = whisper.load_model("base")
    return whisper_model

whisper_model = load_whisper_model()

# Load pre-trained Sentence Transformer model for RAG
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

# Function to load dialogue embeddings
# @st.cache_resource
# def load_dialogue_embeddings():
#     if os.path.exists("dialogue_embeddings.pt"):
#         return torch.load("dialogue_embeddings.pt"), [
#             "The patient has been experiencing headaches for the past week.",
#             "Patient reports a sore throat and fever over the last 3 days.",
#             "Patient has a history of hypertension and is on medication.",
#         ]
#     else:
#         st.error("Dialogue embeddings file not found. Please generate 'dialogue_embeddings.pt' first.")
#         return None, None
@st.cache_resource
def load_dialogue_embeddings():
    if os.path.exists("dialogue_embeddings_with_soap.pt"):
        # Load the dataset to get the original texts
        dataset = load_dataset("omi-health/medical-dialogue-to-soap-summary")
        combined_texts = [
            f"Dialogue: {entry['dialogue']} SOAP Summary: {entry['soap']}"
            for entry in dataset['train']
        ]
        
        # Load the embeddings
        embeddings = torch.load("dialogue_embeddings_with_soap.pt")
        return embeddings, combined_texts
    else:
        st.error("Dialogue embeddings file not found. Please generate 'dialogue_embeddings_with_soap.pt' first.")
        return None, None

def generate_medical_codes(summary):
    messages = [
        {
            "role": "system",
            "content": "You are a Medical Coding Assistant specialized in ICD-10 and CPT codes."
        },
        {
            "role": "user",
            "content": f"""Based on the following medical summary, provide relevant ICD-10 and CPT codes. Format your response as:

            ICD-10 Codes:
            - Code: Description
            
            CPT Codes:
            - Code: Description
            
            Summary: {summary}"""
        }
    ]

    payload = {
        "model": "llama3-70b-8192",
        "messages": messages
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            codes = response.json().get('choices', [{}])[0].get('message', {}).get('content', 'No codes available.')
            return codes
        else:
            st.error(f"Error calling Groq API: {response.status_code} {response.text}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def llama_summarize(text, patient_name="", medical_context=""):
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    messages = [
        {
            "role": "system",
            "content": "You are a Medical Assistant."
        },
        {
            "role": "user",
            "content": f"""Summarize the following conversation,create a record of the patient's visit to the doctor. Format the summary as :

        Subjective:
        - Chief complaint
        - History of present illness
        - Past medical history
        - Review of systems
        
        Objective:
        - Vital signs
        - Physical examination findings
        - Relevant test results
        
        Assessment:
        - Primary diagnosis/impression
        - Differential diagnoses
        - Clinical reasoning
        
        Plan:
        - Diagnostic tests ordered
        - Treatments prescribed
        - Patient education
        - Follow-up instructions

        Similar Medical Cases:
        {medical_context}

        Patient: {patient_name}
        Date: {current_date}
        Conversation: {text}
        """
        }
    ]

    payload = {
        "model": "llama3-70b-8192",
        "messages": messages
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            summary = response.json().get('choices', [{}])[0].get('message', {}).get('content', 'No summary available.')
            return summary
        else:
            st.error(f"Error calling Groq API: {response.status_code} {response.text}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# def retrieve_similar_dialogues(text, dialogue_embeddings, dialogues):
#     query_embedding = embedder.encode(text, convert_to_tensor=True)
#     similarities = torch.nn.functional.cosine_similarity(query_embedding, dialogue_embeddings)
#     top_matches = torch.topk(similarities, k=min(3, len(dialogues))).indices
#     return "\n".join([dialogues[i] for i in top_matches if i < len(dialogues)])
def retrieve_similar_dialogues(text, dialogue_embeddings, dialogues, k=3):
    query_embedding = embedder.encode(text, convert_to_tensor=True)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, dialogue_embeddings)
    top_k = min(k, len(dialogues))
    top_matches = torch.topk(similarities, k=top_k)
    
    similar_cases = []
    for i, index in enumerate(top_matches.indices):
        if index < len(dialogues):
            score = top_matches.values[i]
            similar_cases.append(f"Similar Case {i+1} (Similarity: {score:.2f}):\n{dialogues[index]}\n")
    
    return "\n".join(similar_cases)

def llama_summarize_with_rag(text, patient_name, dialogue_embeddings, dialogues):
    medical_context = retrieve_similar_dialogues(text, dialogue_embeddings, dialogues) if dialogue_embeddings is not None and dialogues else ""
    summary = llama_summarize(text, patient_name, medical_context)
    return summary

def convert_to_wav(audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio = AudioSegment.from_file(audio_data)
        audio.export(temp_audio.name, format="wav")
        return temp_audio.name

def audio_to_text(audio_path):
    """Modified transcription function with content filtering"""
    result = whisper_model.transcribe(audio_path)
    raw_text = result["text"]
    
    # Apply content filtering
    filtered_text, removed_items = filter_sensitive_content(raw_text)
    
    return filtered_text

def update_summary():
    # Update the current summary with the edited text
    st.session_state.current_summary = st.session_state.summary_input

def process_audio(audio_data):
    with st.spinner("Processing audio..."):
        audio_data.seek(0)
        audio_path = convert_to_wav(audio_data)

        # Get filtered transcription
        st.session_state.transcription = audio_to_text(audio_path)

        # Only proceed with summary if there's relevant medical content
        if st.session_state.transcription.strip():
            # Load dialogue embeddings
            dialogue_embeddings, dialogues = load_dialogue_embeddings()

            # Generate summary
            st.session_state.generated_summary = llama_summarize_with_rag(
                st.session_state.transcription,
                st.session_state.patient_name,
                dialogue_embeddings,
                dialogues
            )
            st.session_state.current_summary = st.session_state.generated_summary

            # Generate medical codes
            st.session_state.medical_codes = generate_medical_codes(st.session_state.generated_summary)

            # Change page
            set_page('results')
        else:
            st.error("No relevant medical conversation was found in the recording. Please ensure the recording contains medical-related discussion.")
            return

def main_page():
    st.title("Medical Conversation Summarizer")
    
    # Patient name input
    st.session_state.patient_name = st.text_input("Enter the patient's name:", value=st.session_state.patient_name)

    # Live recording section
    st.subheader("Record Medical Conversation")
    st.write("Click the button below to start recording.")
    wav_audio_data = st.experimental_audio_input("Record Audio", key="audio_recording")

    if wav_audio_data is not None:
        if st.button("Transcribe and Summarize"):
            process_audio(wav_audio_data)
            st.rerun()

def results_page():
    st.title(f"Results for {st.session_state.patient_name}")
    
    # Create tabs for Transcription and Summary
    tab1, tab2 = st.tabs(["Transcription", "Summary"])
    
    # Transcription Section
    with tab1:
        st.subheader("Transcription")
        transcription = st.text_area(
            "Edit transcription if needed:",
            value=st.session_state.transcription,
            height=400,
            key="transcription_input"
        )
        
        # Icons for Copy, Save, and Download
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 Copy Transcription", key="copy_transcription"):
                import pyperclip
                pyperclip.copy(transcription)
                st.success("Transcription copied to clipboard!")
        with col2:
            if st.button("💾 Save Transcription", key="save_transcription"):
                st.session_state.transcription = transcription
                st.success("Transcription saved!")
        with col3:
            st.download_button(
                "⬇️ Download Transcription",
                transcription,
                file_name="transcription.txt",
                mime="text/plain"
            )
    
    # Summary Section
    with tab2:
        st.subheader("Summary")
        summary = st.text_area(
            "Edit summary if needed:",
            value=st.session_state.current_summary,
            height=400,
            key="summary_input"
        )
        
        # Icons for Copy, Save, and Download
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 Copy Summary", key="copy_summary"):
                import pyperclip
                pyperclip.copy(summary)
                st.success("Summary copied to clipboard!")
        with col2:
            if st.button("💾 Save Summary", key="save_summary"):
                update_summary()
                st.success("Summary saved!")
                # Generate new medical codes based on updated summary
                st.session_state.medical_codes = generate_medical_codes(st.session_state.current_summary)
        with col3:
            st.download_button(
                "⬇️ Download Summary",
                summary,
                file_name="patient_summary.txt",
                mime="text/plain"
            )
        
        # Medical Codes Section
        st.subheader("Medical Codes")
        medical_codes = st.text_area(
            "Edit ICD-10 and CPT codes if needed:",
            value=st.session_state.medical_codes if st.session_state.medical_codes else "No codes available.",
            height=200,
            key="medical_codes_input"
        )
        
        # Icons for Copy, Save, and Download
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 Copy Medical Codes", key="copy_medical_codes"):
                import pyperclip
                pyperclip.copy(medical_codes)
                st.success("Medical codes copied to clipboard!")
        with col2:
            if st.button("💾 Save Medical Codes", key="save_medical_codes"):
                st.session_state.medical_codes = medical_codes
                st.success("Medical codes saved!")
        with col3:
            st.download_button(
                "⬇️ Download Medical Codes",
                medical_codes,
                file_name="medical_codes.txt",
                mime="text/plain"
            )
    
    # Back button
    if st.button("Back to Main Page"):
        set_page('main')
        st.rerun()

def filter_sensitive_content(text):
    """
    Filter out sensitive information and non-medical conversation from transcribed text.
    Completely removes sentences containing sensitive information.
    """
    # Patterns for sensitive information
    sensitive_patterns = {
        'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
        'phone': r'\b(?:\+?1[-.]?)?\s*(?:\([0-9]{3}\)|[0-9]{3})[-.\s][0-9]{3}[-.\s][0-9]{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'ssn': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        'address': r'\b\d+\s+([A-Za-z]+ ){1,2}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
        'dob': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2})\b'
    }
    
    # Medical keywords for relevance filtering
    medical_keywords = [
        'pain', 'symptom', 'doctor', 'medication', 'treatment', 'health',
        'medical', 'hospital', 'clinic', 'prescription', 'disease', 'condition',
        'patient', 'diagnosis', 'test', 'exam', 'history', 'allergies', 'surgery',
        'blood', 'heart', 'breathing', 'fever', 'chronic', 'acute', 'follow-up',
        'diet', 'exercise', 'weight', 'pressure', 'temperature', 'pulse',
        'symptoms', 'medications', 'allergic', 'family history', 'medical history',
        'side effects', 'dosage', 'recovery', 'appointment', 'specialist',
        'referral', 'insurance', 'prescription', 'laboratory', 'results'
    ]

    # Split text into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    filtered_sentences = []
    filtered_items = []

    for sentence in sentences:
        # Skip empty sentences
        if not sentence:
            continue

        # Check if sentence contains any sensitive information
        contains_sensitive_info = any(
            re.search(pattern, sentence, re.IGNORECASE)
            for pattern in sensitive_patterns.values()
        )

        # Skip sentence if it contains sensitive information
        if contains_sensitive_info:
            filtered_items.append('sensitive_information')
            continue

        # Check for medical relevance
        is_medical = any(keyword.lower() in sentence.lower() for keyword in medical_keywords)
        is_greeting = bool(re.search(r'\b(hello|hi|good morning|good afternoon)\b', sentence.lower()))
        is_farewell = bool(re.search(r'\b(goodbye|bye|thank you|thanks|see you)\b', sentence.lower()))
        
        # Only include relevant sentences
        if is_medical or is_greeting or is_farewell:
            filtered_sentences.append(sentence)
        else:
            filtered_items.append('off_topic_conversation')

    # Reconstruct the filtered text
    filtered_text = '. '.join(filtered_sentences)
    if filtered_text and not filtered_text.endswith('.'):
        filtered_text += '.'

    return filtered_text, list(set(filtered_items))

# Main app logic

def main():
    if st.session_state.page == 'main':
        main_page()
    elif st.session_state.page == 'results':
        results_page()

if __name__ == "__main__":
    main()