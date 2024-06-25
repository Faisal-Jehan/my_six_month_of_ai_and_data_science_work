import streamlit as st

def main():
    st.title("My First Streamlit App")
    
    # Text input for user's name
    name = st.text_input("Enter your name:")
    
    # Button to trigger the greeting
    if st.button("Greet"):
        if name:
            st.write(f"Hello, {name}! Welcome to your first Streamlit app.")
        else:
            st.write("Hello! Welcome to your first Streamlit app.")

if __name__ == "__main__":
    main()

