import streamlit as st
import pickle
import re
import math

# Load the trained model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)


# Convert message into words
def get_words(message):
    return re.findall(r'\b\w+\b', message.lower())


# Predict the message
def predict(message):

    words = get_words(message)

    spam_words = model["spam_words"]
    normal_words = model["normal_words"]

    spam_count = model["spam_count"]
    normal_count = model["normal_count"]

    total_spam_words = model["total_spam_words"]
    total_normal_words = model["total_normal_words"]

    spam_score = math.log(spam_count / (spam_count + normal_count))
    normal_score = math.log(normal_count / (spam_count + normal_count))

    vocabulary = set(spam_words) | set(normal_words)
    vocabulary_size = len(vocabulary)

    for word in words:

        spam_probability = (
            spam_words.get(word, 0) + 1
        ) / (total_spam_words + vocabulary_size)

        normal_probability = (
            normal_words.get(word, 0) + 1
        ) / (total_normal_words + vocabulary_size)

        spam_score += math.log(spam_probability)
        normal_score += math.log(normal_probability)

    if spam_score > normal_score:
        return "SPAM"

    return "NORMAL"


# -----------------------------
# Streamlit User Interface
# -----------------------------

st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📱"
)

st.title("📱 Smart Spam Message Detector")

st.write(
    "Enter an SMS or message below and the AI will classify it."
)

message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You won a free prize!"
)


if st.button("🔍 Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        result = predict(message)

        if result == "SPAM":
            st.error("🚨 SPAM MESSAGE")
            st.write("This message may be suspicious.")

        else:
            st.success("✅ NORMAL MESSAGE")
            st.write("This message looks normal.")
