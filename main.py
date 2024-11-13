import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download necessary nltk resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load vectorizer and model with error handling
try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()  # Stop further execution if files cannot be loaded

# Streamlit app interface
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

# Button for prediction
if st.button('Predict'):
    if input_email == "":
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess the text
        transformed_email = transform_text(input_email)
        st.write(f"Transformed Text: {transformed_email}")

        # 2. Vectorize the input
        vector_input = tfidf.transform([transformed_email])

        # 3. Predict with the model
        result = model.predict(vector_input)[0]

        # 4. Display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
