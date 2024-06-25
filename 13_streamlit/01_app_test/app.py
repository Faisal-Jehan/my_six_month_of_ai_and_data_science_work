import streamlit as st

# Adding title of your app
st.title('ZARAB app ')

# adding simple text
st.write('Here is a simple text')

# user input
number = st.slider('Pick a number', 0, 100, 20)

st.write(f'You selected: {number}')


# adding a button
if st.button('Greetings'):
    st.write('hi, hello there')

else:
    st.write('Goodbye')

# add radio button with options
genre = st.radio(
    "What is your favourite movie genre",
    ('Comedy', 'Drama', 'Documentary'))

# print the text of genre
st.write(f'You selected: {genre}')

# add a drop down list
#options = st.selectbox(
 #   'How would you like to be contacted?',
  #  ('Email', 'home phone', 'Mobile phone'))


# add a drop down list on the left slidebar
option = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'home phone', 'Mobile phone'))

# add your whatsapp number 
st.sidebar.text_input('Enter your whatsapp number')

# add a file uploader 
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")