import streamlit as st
import re
import pickle
from scipy.sparse import hstack

# Load model and vectorizers
@st.cache_resource
def load_model_and_vectorizers():
    with open("SGD_Model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("Char_vector.pkl", "rb") as f:
        char_vectorizer = pickle.load(f)
    with open("Word_vector.pkl", "rb") as f:
        word_vectorizer = pickle.load(f)
    return model, word_vectorizer, char_vectorizer

model, word_vectorizer, char_vectorizer = load_model_and_vectorizers()


def clean_processing(text):
    """Clean and preprocess the input text."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)          # Remove HTML tags
    text = re.sub(r"http\S+", " ", text)        # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text) # Keep letters, numbers, apostrophes
    text = re.sub(r"\s+", " ", text).strip()    # Remove extra spaces
    return text


# Streamlit UI
st.title("Sentiment Classifier")
st.markdown("Enter text to determine whether the sentiment is **positive** or **negative**.")

text_input = st.text_area(
    "Enter your text:", 
    height=200,
    placeholder="Paste the text here..."
)

if st.button("Check Sentiment"):
    if not text_input or not text_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Processing..."):
            # Preprocess
            cleaned_text = clean_processing(text_input)
            
            if not cleaned_text:
                st.error("Unable to process the provided text.")
            else:
                # Transform using vectorizers (use transform, not fit_transform)
                X_word = word_vectorizer.transform([cleaned_text])
                X_char = char_vectorizer.transform([cleaned_text])
                
                # Combine word and character features
                X = hstack([X_word, X_char])
                
                # Predict
                prediction = model.predict(X)[0]
                
                # Display result
                if prediction == 1:
                    st.success("✅ The text is classified as **Positive**.")
                else:
                    st.error("❌ The text is classified as **Negative**.")
                
                # Optional: Show confidence/probability
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]
                    st.info(f"Confidence: Positive = {proba[0]:.2%}, Negative = {proba[1]:.2%}")

# Sidebar information
st.sidebar.header("About")
st.sidebar.info(
    "This application uses an SGD classifier with combined word and character n-gram features.\n\n"
    "Text is cleaned and transformed using pre-fitted vectorizers before prediction."
)