import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

st.title("🔧 Debug Mode - Model Investigation")

# Step 1: Check if model file exists and what's inside
try:
    st.write("**Step 1: Checking model file...**")
    checkpoint = torch.load('conditional_vae_mnist.pth', map_location='cpu')
    st.success("✅ Model file loaded successfully!")
    
    # Show what's in the checkpoint
    st.write("**Checkpoint keys:**", list(checkpoint.keys()))
    
    if 'model_state_dict' in checkpoint:
        st.write("**Model state dict keys:**")
        model_keys = list(checkpoint['model_state_dict'].keys())
        for key in model_keys:
            st.write(f"- {key}")
        
        st.write(f"**Total layers found:** {len(model_keys)}")
    
    if 'model_config' in checkpoint:
        st.write("**Model config:**", checkpoint['model_config'])
    
except Exception as e:
    st.error(f"❌ Error loading model file: {e}")
    st.stop()

# Step 2: Try to create the model architecture
st.write("**Step 2: Creating model architecture...**")

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(784 + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

try:
    model = ConditionalVAE(latent_dim=20, num_classes=10)
    st.success("✅ Model architecture created successfully!")
    
    # Show model structure
    st.write("**Model layers we created:**")
    for name, param in model.named_parameters():
        st.write(f"- {name}: {param.shape}")
    
except Exception as e:
    st.error(f"❌ Error creating model: {e}")
    st.stop()

# Step 3: Try to load the state dict
st.write("**Step 3: Attempting to load state dict...**")

try:
    # Get the saved state dict keys
    saved_keys = set(checkpoint['model_state_dict'].keys())
    model_keys = set(dict(model.named_parameters()).keys())
    
    st.write("**Comparison:**")
    st.write(f"- Keys in saved model: {len(saved_keys)}")
    st.write(f"- Keys in our model: {len(model_keys)}")
    
    # Find missing keys
    missing_in_saved = model_keys - saved_keys
    missing_in_model = saved_keys - model_keys
    
    if missing_in_saved:
        st.error("**❌ Keys in our model but NOT in saved model:**")
        for key in missing_in_saved:
            st.write(f"- {key}")
    
    if missing_in_model:
        st.error("**❌ Keys in saved model but NOT in our model:**")
        for key in missing_in_model:
            st.write(f"- {key}")
    
    if not missing_in_saved and not missing_in_model:
        st.success("✅ All keys match! Attempting to load...")
        model.load_state_dict(checkpoint['model_state_dict'])
        st.success("✅ Model loaded successfully!")
        
        # Test generation
        st.write("**Step 4: Testing generation...**")
        with torch.no_grad():
            labels = torch.full((2,), 0, dtype=torch.long)
            z = torch.randn(2, 20)
            y_onehot = F.one_hot(labels, num_classes=10).float()
            z_labeled = torch.cat([z, y_onehot], dim=1)
            generated = model.decoder(z_labeled)
            st.success("✅ Generation test passed!")
            
            # Show a simple test image
            test_img = generated[0].view(28, 28)
            test_img = (test_img + 1) / 2
            test_img = (test_img.numpy() * 255).astype('uint8')
            st.image(test_img, caption="Test generated image", width=100)
    else:
        st.error("❌ Key mismatch prevents loading the model!")
        
except Exception as e:
    st.error(f"❌ Error during state dict loading: {e}")
    st.write(f"**Full error details:** {str(e)}")
