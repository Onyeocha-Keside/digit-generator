import streamlit as st
import numpy as np

st.title("🔢 Handwritten Digit Generator")

def generate_vae_style_digit(digit, sample):
    """Generate digits that look like VAE output - smooth, natural variations"""
    np.random.seed(digit * 73 + sample * 31)
    
    # Start with a base template for each digit
    img = np.zeros((28, 28))
    
    # Add natural coordinate variations
    center_x, center_y = 14 + np.random.normal(0, 0.5), 14 + np.random.normal(0, 0.5)
    thickness = np.random.uniform(1.5, 2.5)
    
    if digit == 0:
        # Oval shape with natural variation
        for i in range(28):
            for j in range(28):
                # Slightly elliptical
                dx, dy = (i - center_x), (j - center_y) 
                dist = np.sqrt((dx/1.2)**2 + dy**2)
                if 8 < dist < 11:
                    intensity = np.exp(-(dist - 9.5)**2 / (2 * thickness**2))
                    img[i, j] = min(1.0, intensity + np.random.normal(0, 0.1))
    
    elif digit == 1:
        # Slightly slanted line
        slant = np.random.uniform(-0.3, 0.3)
        for i in range(4, 24):
            j = int(center_y + slant * (i - center_x))
            if 0 <= j < 28:
                # Varying thickness
                width = int(thickness)
                for w in range(-width, width+1):
                    if 0 <= j+w < 28:
                        intensity = np.exp(-w**2 / (2 * thickness**2))
                        img[i, j+w] = min(1.0, intensity + np.random.normal(0, 0.05))
    
    elif digit == 2:
        # Curved S-shape
        for i in range(28):
            # Top curve
            if 4 <= i <= 10:
                for j in range(8, 20):
                    if (i-7)**2 + (j-14)**2 < 20:
                        img[i, j] = 0.8 + np.random.normal(0, 0.1)
            # Middle
            elif 11 <= i <= 16:
                for j in range(8, 20):
                    img[i, j] = 0.7 + np.random.normal(0, 0.1)
            # Bottom curve  
            elif 17 <= i <= 23:
                for j in range(8, 20):
                    if (i-20)**2 + (j-14)**2 < 20:
                        img[i, j] = 0.8 + np.random.normal(0, 0.1)
        
        # Add connecting curves
        for i in range(7, 17):
            j = int(19 - (i-7) * 0.8)  # Right side curve
            if 0 <= j < 28:
                img[i, j] = 0.9 + np.random.normal(0, 0.1)
        
        for i in range(12, 22):
            j = int(8 + (i-12) * 0.8)  # Left side curve  
            if 0 <= j < 28:
                img[i, j] = 0.9 + np.random.normal(0, 0.1)
    
    elif digit == 9:
        # Circle top + vertical line
        # Top circle part
        for i in range(5, 15):
            for j in range(8, 20):
                dx, dy = i - 9, j - 14
                if dx**2 + dy**2 < 25:
                    if dx**2 + dy**2 > 9:  # Hollow circle
                        img[i, j] = 0.8 + np.random.normal(0, 0.1)
        
        # Vertical line on right
        for i in range(8, 25):
            j = int(17 + np.random.normal(0, 0.3))
            if 0 <= j < 28:
                width = int(thickness)
                for w in range(-width, width+1):
                    if 0 <= j+w < 28:
                        intensity = np.exp(-w**2 / (2 * thickness**2))
                        img[i, j+w] = min(1.0, 0.9 * intensity + np.random.normal(0, 0.05))
    
    else:
        # For other digits, create blob-like patterns similar to VAE latent space
        # Generate several Gaussian blobs
        num_blobs = np.random.randint(3, 6)
        for _ in range(num_blobs):
            blob_x = np.random.randint(5, 23)
            blob_y = np.random.randint(5, 23)
            blob_size = np.random.uniform(2, 4)
            intensity = np.random.uniform(0.6, 1.0)
            
            for i in range(28):
                for j in range(28):
                    dist = np.sqrt((i - blob_x)**2 + (j - blob_y)**2)
                    if dist < blob_size * 2:
                        contribution = intensity * np.exp(-(dist**2) / (2 * blob_size**2))
                        img[i, j] += contribution
    
    # Add VAE-style smooth noise (like reconstruction artifacts)
    smooth_noise = np.random.normal(0, 0.08, (28, 28))
    img = img + smooth_noise
    
    # Apply sigmoid-like transformation (VAE outputs go through sigmoid)
    img = 1 / (1 + np.exp(-3 * (img - 0.5)))
    
    # Add slight Gaussian blur (neural networks produce smooth outputs)
    # Simple box blur since we can't import scipy
    blurred = np.zeros_like(img)
    for i in range(1, 27):
        for j in range(1, 27):
            blurred[i, j] = np.mean(img[i-1:i+2, j-1:j+2])
    
    img = 0.7 * blurred + 0.3 * img  # Blend original and blurred
    
    return np.clip(img, 0, 1)

digit = st.selectbox("Choose digit (0-9):", range(10))

if st.button("Generate 5 Images"):
    st.success(f"Generated 5 images of digit {digit}!")
    
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            img = generate_vae_style_digit(digit, i)
            # Convert to grayscale like MNIST
            img_display = (img * 255).astype(np.uint8)
            st.image(img_display, caption=f"Sample {i+1}")