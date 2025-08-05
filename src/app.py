import gradio as gr
from predict import predict_spam

demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(label="Enter a message"),
    outputs=gr.Label(label="Prediction"),
    title="📨 Spam Detector",
    description="A simple spam detection demo using a trained model"
)

demo.launch()
