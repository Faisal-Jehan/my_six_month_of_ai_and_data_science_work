# create a text to speech app
import streamlit as st
from openai import OpenAI
import tempfile
import os

# Function to convert text to speech, modified to explicitly use an API key
def text_to_speech(api_key, text: str):
    """
    Converts text to speech using OpenAI's tts-1 model and saves the output as an MP3 file,
    explicitly using an API key for authentication.
    """

    # Initialize the OpenAI client with the provided API key
    client = OpenAI(api_key=api_key)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        speech_file_path = tmpfile.name
        response = client.audio.speech.create(
            model = model,
            voice = voice,
            input=text
        )
        # Stream the audio response to file 
        response.stream_to_file(speech_file_path)

        # Return the path to the audio file
        return speech_file_path
    
# Streamlit UserInterface(UI) Seteuo
st.title("🔊 Text to Speech Converter 📝")
st.image("https://www.piecex.com/product_image/20190625044028-00000544-image2.png")
st.markdown("""
This app converts text to speech(tts) using OpenAI's tts-1 model. 
Please enter your OpenAI API key below. **Do not share your API key with others.**
""")

# Input for OpenAI API key 
api_key = st.text_input("Enter your OpenAI API key", type="password")

# create a select box for a model 
model = st.selectbox("Select a model", ["tts-1", "tts-1-hd"])

# create a select box for the vocal from these alloy, echo, fable, onyx, nova and shimmer
voice = st.selectbox("Select a vocal", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])

# Text input from user 
user_input = st.text_area("Enter text that convert into speech", "Hello, welcome to our text to speech converter!")

if st.button("Convert"):
    if not api_key:
        st.error("API key is required to convert text to speech.")
    else:
        try:
            speech_path = text_to_speech(api_key, user_input)

            # Display a link to download the MP3 file 
            st.audio(open(speech_path, 'rb'), format="audio/mp3")

            # Add a download button 
            with open(speech_path, "rb") as file:
                st.download_button(
                    label="Download MP3",
                    data = file,
                    file_name = "speech.mp3",
                    mime = "audio/mpeg"
                )

            # Clean up: delete the temporary file after use
            os.remove(speech_path)
        except Exception as e:
            st.error(f"An error occurred: {e}")



# # Addih=ng the HTML footer
# # Profile footer HTML

# # create a function to display the voice options
# def display_text(text):
#     st.write(text)

# # create a function to display the voice options from these options:alloy, echo, fable, onyx, nova, and shimmer
# def display_voice_options():
#     voice = st.selectbox("Select a voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
#     return voice

# # create a function to display the text input
# def display_text_input():
#     text = st.text_area("Enter the text to convert to speech")
#     return text 

# # create a function to display the convert button 
# def display_convert_button():
#     if st.button("Convert"):
#         return True
#     return False

# # create a function to display the main app
# def main():
#     st.title("Text to Speech App")
#     text = display_text_input()
#     voice = display_voice_options()
#     convert = display_convert_button()
#     if convert:
#         audio_file = text_to_speech(text, voice)
#         display_audio(audio_file)

# if __name__ == "__main__":
#     main()







