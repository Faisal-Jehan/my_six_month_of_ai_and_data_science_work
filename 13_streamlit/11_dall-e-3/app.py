# Import libraries
import streamlit as st
import openai
import requests
from io import BytesIO

def generate_image(api_key, prompt):
    openai.api_key = api_key
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt,
        n=1,  # Number of images to generate
        size="1024x1024"  # Image size, can adjust based on your requirement
    )
    return response

def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return BytesIO(response.content)
    return None

def main():
    st.title('DALL-E 3 Image Generator')

    api_key = st.text_input('Enter your OpenAI API key:', type='password')
    prompt = st.text_input('Enter a prompt for DALL-E 3:', '')

    if st.button('Generate Image'):
        if api_key:
            with st.spinner('Generating Image...'):
                response = generate_image(api_key, prompt)
                if response and 'data' in response and len(response['data']) > 0:
                    image_url = response['data'][0]['url']
                    st.image(image_url, caption='Generated Image', use_column_width=True)

                    # Download functionality
                    image_buffer = download_image(image_url)
                    if image_buffer:
                        st.download_button(
                            label="Download Image",
                            data=image_buffer,
                            file_name="generated_image.png",
                            mime="image/png"
                        )
                    else:
                        st.error("Failed to download image.")
                else:
                    st.error('Failed to generate image. Please try again.')
        else:
            st.error('Please enter your OpenAI API key.')

if __name__ == '__main__':
    main()
