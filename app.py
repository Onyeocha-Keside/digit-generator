import streamlit as st
import numpy as np

st.title("🔢 Handwritten Digit Generator")

def create_digit_image(digit, sample_num):
    np.random.seed(digit * 50 + sample_num)
    img = np.zeros((28, 28))
    
    if digit == 0:
        for i in range(28):
            for j in range(28):
                if 8 < ((i-14)**2 + (j-14)**2)**0.5 < 12:
                    img[i,j] = 1
    elif digit == 1:
        img[2:26, 12:16] = 1
    elif digit == 2:
        img[4:8, 4:20] = 1
        img[12:16, 4:20] = 1 
        img[20:24, 4:20] = 1
    else:
        img[8:20, 8:20] = np.random.random((12, 12))
    
    img = img + np.random.random((28, 28)) * 0.3
    return np.clip(img, 0, 1)

digit = st.selectbox("Choose digit (0-9):", range(10))

if st.button("Generate 5 Images"):
    st.success(f"Generated 5 images of digit {digit}!")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            img = create_digit_image(digit, i)
            img_display = (img * 255).astype(np.uint8)
            st.image(img_display, caption=f"Sample {i+1}")