"""
EcoSense AI Server - Hugging Face Gradio & FastAPI Web App
"""
import gradio as gr
import numpy as np
from ai.predictor import get_predictor

# Load Singleton predictor ONCE
predictor = get_predictor()

def classify_audio_file(audio_path):
    """Classify uploaded audio file and return species prediction."""
    if not audio_path:
        return "Please upload an audio file (.wav, .mp3, .flac)."
    
    result = predictor.predict(audio_path)
    if result.get("status") == "error":
        return f"Error: {result.get('message')}"
    
    pred = result.get("prediction", {})
    common_name = pred.get("common_name", "Unknown")
    sci_name = pred.get("scientific_name", "Unknown")
    confidence = pred.get("confidence", 0.0)
    
    top_preds = result.get("top_predictions", [])
    top_str = "\n".join([f"• {p['common_name']} ({p['scientific_name']}): {p['confidence']}%" for p in top_preds[:5]])
    
    return f"🦜 Identified Species: {common_name}\n🔬 Scientific Name: {sci_name}\n🎯 Confidence: {confidence}%\n\nTop 5 Matches:\n{top_str}"

# Create Gradio UI
demo = gr.Interface(
    fn=classify_audio_file,
    inputs=gr.Audio(type="filepath", label="Upload Bird Sound Audio Clip"),
    outputs=gr.Textbox(label="AI Detection Result", lines=8),
    title="EcoSense AI - Bird Sound Classifier",
    description="Official Production AI Inference Model (YAMNet + Keras Classifier across 66 Species)",
    theme="soft",
)

# Launch app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
