# i want to create a complete streamlit app for speech to text
import streamlit as st
from openai import OpenAI
import tempfile
import os 

# SideBar for API key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Main app title
st.title("🎙️ Speech To Text using Whisper-1 🤖")

# Upload audio file 
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

# Initialize the OpenAI client with the API key from the sidebar
client = OpenAI(api_key=api_key)

if audio_file is not None and api_key:
    # Save the uploaded file temporaily
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + audio_file.name.split('.')[-1]) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Open the saved audio file in binary read mode
        with open(tmp_file_path, "rb") as audio_file:
            # Transcribe the audio file using OpenAI's whisper model
            transcription_response = client.audio.transcriptions.create(
                model = "whisper-1",
                file = audio_file
            )

        # According the transcription text correctly
        transcription_text = transcription_response.text

        # Display the transcription text correctly
        st.write("Transcription:", transcription_text)

    except Exception as e:
        # Display any errors that occur during transcription
        st.error(f"An error occurred: {str(e)}")
        
    finally:
        # Clean up: Remove the temoorary file
        os.remove(tmp_file_path)


           

