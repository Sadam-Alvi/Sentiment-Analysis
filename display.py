import streamlit as st
import re
import pickle
from sentence_transformers import SentenceTransformer


# Load BGE embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        device="cpu"
    )


embeddings = load_embedding_model()


# Load SGD model
@st.cache_resource
def load_model():
    with open("SGD_Model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()


def clean_processing(text):
    """Clean and preprocess the input text."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Streamlit UI
st.title("Comment's Sentiment Classifier")

st.markdown(
    "Enter text to determine whether the sentiment is "
    "**positive** or **negative**."
)


# Example texts
positive_example = (
    "I absolutely love this product! It works perfectly "
    "and exceeded all my expectations. Highly recommended."
)

negative_example = (
    "This is the worst experience I've ever had. "
    "The product broke immediately and customer service was terrible."
)


# Example buttons
st.markdown("**Try an example:**")

col1, col2 = st.columns(2)

with col1:
    if st.button("😊 Positive Example"):
        st.session_state["user_text"] = positive_example

with col2:
    if st.button("😞 Negative Example"):
        st.session_state["user_text"] = negative_example


# Initialize session state
if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""


text_input = st.text_area(
    "Enter your text:",
    value=st.session_state["user_text"],
    height=200,
    placeholder="Paste the text here..."
)


# Prediction
if st.button("Check Sentiment"):

    if not text_input or not text_input.strip():

        st.warning("Please enter some text to classify.")

    else:

        with st.spinner("Processing..."):

            # Clean text
            cleaned_text = clean_processing(text_input)

            if not cleaned_text:

                st.error("Unable to process the provided text.")

            else:

                # Convert text to BGE embedding
                X = embeddings.encode(
                    [cleaned_text],
                    normalize_embeddings=True
                )

                # Predict sentiment
                prediction = model.predict(X)[0]

                # Display result
                if prediction == 1:
                    st.success(
                        "✅ The text is classified as **Positive**."
                    )
                else:
                    st.error(
                        "❌ The text is classified as **Negative**."
                    )

                # Confidence
                if hasattr(model, "predict_proba"):

                    proba = model.predict_proba(X)[0]

                    st.info(
                        f"Confidence: Positive = {proba[1]:.2%}, "
                        f"Negative = {proba[0]:.2%}"
                    )


# Sidebar
st.sidebar.header("About")

st.sidebar.info(
    "This application uses an SGD classifier with "
    "BAAI/bge-small-en-v1.5 embeddings.\n\n"
    "Text is cleaned and converted into semantic embeddings "
    "before prediction."
)