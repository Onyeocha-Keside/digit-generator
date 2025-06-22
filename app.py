import streamlit as st
import numpy as np
from PIL import Image

def generate_digit_pattern(digit, sample_num):
    """Generate a simple digit-like pattern"""
    np.random.seed(digit * 100 + sample_num)  # Different seed for each sample
    
    # Create base 28x28 image
    img = np.zeros((28, 28))
    
    # Simple patterns for each digit
    if digit == 0:
        # Circle pattern
        center = (14, 14)
        for i in range(28):
            for j in range(28):
                dist = ((i - center[0])**2 + (j - center[1])**2)**0.5
                if 8 < dist < 12:
                    img[i, j] = 1
    
    elif digit == 1:
        # Vertical line
        img[4:24, 12:16] = 1
    
    elif digit == 2:
        # S-curve pattern
        img[4:8, 8:20] = 1
        img[12:16, 8:20] = 1
        img[20:24, 8:20] = 1
        img[8:12, 16:20] = 1
        img[16:20, 8:12] = 1
    
    else:
        # Generic pattern for other digits
        img[6:22, 6:22] = np.random.random((16, 16)) > 0.7
    
    # Add random noise for variation
    noise = np.random.random((28, 28)) * 0.3
    img = img + noise
    img = np.clip(img, 0, 1)
    
    return (img * 255).astype(np.uint8)

st.title("🔢 Handwritten Digit Generator")
st.write("Generate 5 unique handwritten-style digits")

digit = st.selectbox("Choose digit (0-9):", range(10))

if st.button("Generate 5 Images"):
    st.success(f"Generated 5 images of digit {digit}!")
    
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            img = generate_digit_pattern(digit, i)
            st.image(img, caption=f"Sample {i+1}")
