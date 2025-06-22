import streamlit as st
import numpy as np

st.title("🔢 Handwritten Digit Generator")

def create_clear_digit(digit, sample):
    np.random.seed(digit * 137 + sample * 23)
    img = np.zeros((28, 28))
    
    if digit == 0:
        # Circle
        for i in range(28):
            for j in range(28):
                dist = ((i - 14)**2 + (j - 14)**2)**0.5
                if 9 <= dist <= 12:
                    img[i, j] = 1.0
    
    elif digit == 1:
        # Vertical line
        img[3:25, 12:16] = 1.0
    
    elif digit == 2:
        # Number 2 shape
        img[4:7, 8:20] = 1.0      # Top horizontal
        img[12:15, 8:20] = 1.0    # Middle horizontal
        img[21:24, 8:20] = 1.0    # Bottom horizontal
        img[7:12, 17:20] = 1.0    # Top right vertical
        img[15:21, 8:11] = 1.0    # Bottom left vertical
    
    elif digit == 3:
        # Number 3 shape
        img[4:7, 8:20] = 1.0      # Top horizontal
        img[12:15, 8:18] = 1.0    # Middle horizontal
        img[21:24, 8:20] = 1.0    # Bottom horizontal
        img[7:12, 17:20] = 1.0    # Top right vertical
        img[15:21, 17:20] = 1.0   # Bottom right vertical
    
    elif digit == 4:
        # Number 4 shape
        img[3:15, 6:9] = 1.0      # Left vertical
        img[12:15, 6:20] = 1.0    # Horizontal crossbar
        img[3:25, 17:20] = 1.0    # Right vertical
    
    elif digit == 5:
        # Number 5 shape - THIS WAS THE PROBLEM!
        img[4:7, 8:20] = 1.0      # Top horizontal
        img[12:15, 8:18] = 1.0    # Middle horizontal
        img[21:24, 8:20] = 1.0    # Bottom horizontal
        img[7:12, 8:11] = 1.0     # Top left vertical
        img[15:21, 17:20] = 1.0   # Bottom right vertical
    
    elif digit == 6:
        # Number 6 shape
        img[4:7, 8:18] = 1.0      # Top horizontal
        img[7:21, 8:11] = 1.0     # Left vertical
        img[12:15, 8:18] = 1.0    # Middle horizontal
        img[15:21, 17:20] = 1.0   # Bottom right vertical
        img[21:24, 8:18] = 1.0    # Bottom horizontal
    
    elif digit == 7:
        # Number 7 shape
        img[4:7, 8:20] = 1.0      # Top horizontal
        img[7:24, 16:19] = 1.0    # Diagonal line
    
    elif digit == 8:
        # Number 8 shape
        img[4:7, 9:19] = 1.0      # Top horizontal
        img[7:12, 8:11] = 1.0     # Top left vertical
        img[7:12, 17:20] = 1.0    # Top right vertical
        img[12:15, 9:19] = 1.0    # Middle horizontal
        img[15:21, 8:11] = 1.0    # Bottom left vertical
        img[15:21, 17:20] = 1.0   # Bottom right vertical
        img[21:24, 9:19] = 1.0    # Bottom horizontal
    
    elif digit == 9:
        # Number 9 shape
        img[4:7, 9:19] = 1.0      # Top horizontal
        img[7:15, 8:11] = 1.0     # Top left vertical
        img[7:24, 17:20] = 1.0    # Right vertical
        img[12:15, 9:19] = 1.0    # Middle horizontal
        img[21:24, 9:19] = 1.0    # Bottom horizontal
    
    # Add tiny bit of natural variation
    variation = np.random.random((28, 28)) * 0.1
    img = np.clip(img + variation, 0, 1)
    
    return img

digit = st.selectbox("Choose digit (0-9):", range(10))

if st.button("Generate 5 Images"):
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            img = create_clear_digit(digit, i)
            # Make it crisp black and white
            img_clean = (img > 0.5).astype(np.uint8) * 255
            st.image(img_clean, caption=f"Sample {i+1}")