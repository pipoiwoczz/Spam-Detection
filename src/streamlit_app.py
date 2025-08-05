import streamlit as st
from predict import predict_spam

st.title("📩 Spam Detector")

user_input = st.text_area("Enter a message:")
if st.button("Check"):
    result = predict_spam(user_input)
    st.subheader(f"Prediction: {result}")
