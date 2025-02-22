import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
import torch.nn as nn
from PIL import Image
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Baby Cry Analyzer",
    page_icon="👶",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Load the model from pickle file
    with open("baby_cry_model.pkl", "rb") as f:
        model_state_dict = pickle.load(f)
    
    # Initialize model architecture
    model = vit_b_16(pretrained=True)
    num_classes = 3  # Adjust based on your actual number of classes
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model.load_state_dict(model_state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, device

def create_spectrogram(audio_file):
    # Create spectrogram
    y, sr = librosa.load(audio_file, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel")
    plt.axis("off")
    
    # Save spectrogram
    temp_path = "temp_spectrogram.png"
    plt.savefig(temp_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    
    return temp_path

def classify_audio(model, device, spectrogram_path):
    # Prepare image for classification
    img = Image.open(spectrogram_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = transform(img).unsqueeze(0).to(device)
    
    # Classify
    with torch.no_grad():
        output = model(img)
        predicted_class = torch.argmax(output, dim=1).item()
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    return predicted_class, probabilities

def main():
    st.title("👶 Baby Cry Analyzer")
    st.write("Upload a WAV file to analyze the type of baby cry")
    
    # Load model
    try:
        model, device = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # File upload
    audio_file = st.file_uploader("Choose a WAV file", type=['wav'])
    
    if audio_file is not None:
        st.audio(audio_file)
        
        with st.spinner("Analyzing audio..."):
            # Create and display spectrogram
            spec_path = create_spectrogram(audio_file)
            st.image(spec_path, caption="Generated Spectrogram", width=300)
            
            # Classify
            predicted_class, probabilities = classify_audio(model, device, spec_path)
            
            # Display results
            classes = ['Belly Pain', 'Hungry', 'Tired']  # Adjust based on your classes
            st.subheader("Classification Results:")
            
            # Display prediction with confidence
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Cry Type", classes[predicted_class])
            with col2:
                confidence = float(probabilities[predicted_class]) * 100
                st.metric("Confidence", f"{confidence:.2f}%")
            
            # Show all probabilities
            st.subheader("Probability Distribution:")
            for cls, prob in zip(classes, probabilities):
                st.write(f"{cls}: {float(prob)*100:.2f}%")
            
            # Cleanup
            if os.path.exists(spec_path):
                os.remove(spec_path)

if __name__ == "__main__":
    main()
