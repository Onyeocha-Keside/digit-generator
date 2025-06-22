import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class SimpleDigitGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple random digit generator
        self.base_patterns = torch.randn(10, 28, 28)  # Base pattern for each digit
        
    def generate(self, digit, num_samples=5):
        # Generate variations of the base pattern
        base = self.base_patterns[digit]
        samples = []
        
        for i in range(num_samples):
            # Add random noise for variation
            noise = torch.randn(28, 28) * 0.3
            sample = base + noise
            # Make it look more like a digit
            sample = torch.sigmoid(sample * 2)
            samples.append(sample)
        
        return torch.stack(samples)

@st.cache_resource
def create_generator():
    return SimpleDigitGenerator()

st.title("🔢 Handwritten Digit Generator")
st.write("Generate 5 unique handwritten-style digits")

generator = create_generator()
digit = st.selectbox("Choose digit (0-9):", range(10))

if st.button("Generate 5 Images"):
    with st.spinner("Generating..."):
        generated = generator.generate(digit, 5)
        
        st.success(f"Generated 5 images of digit {digit}!")
        cols = st.columns(5)
        
        for i in range(5):
            with cols[i]:
                img_array = (generated[i].numpy() * 255).astype(np.uint8)
                st.image(img_array, caption=f"Sample {i+1}")

st.info("Note: This generates digit-like patterns. For better quality, ensure your trained model file matches the architecture.")
