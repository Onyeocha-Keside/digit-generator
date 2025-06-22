import streamlit as st
import numpy as np

st.title("🔢 Handwritten Digit Generator")

def generate_digit(digit, sample):
    np.random.seed(digit * 100 + sample)
    img = np.zeros((28, 28))
    
    if digit == 0:
        for i in range(28):
            for j in range(28):
                if 8 < ((i-14)**2 + (j-14)**2)**0.5 < 12:
                    img[i,j] = 1
    elif digit == 1:
        img[2:26, 12:16] = 1
    else:
        img[8:20, 8:20] = np.random.random((12, 12))
    
    return np.clip(img + np.random.random((28, 28)) * 0.3, 0, 1)

digit = st.selectbox("Choose digit (0-9):", range(10))

if st.button("Generate 5 Images"):
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            img = generate_digit(digit, i)
            st.image((img * 255).astype('uint8'), caption=f"Sample {i+1}")