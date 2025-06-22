import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784 + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Latent space
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def decode(self, z, y):
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        z_labeled = torch.cat([z, y_onehot], dim=1)
        return self.decoder(z_labeled)

@st.cache_resource
def load_model():
    checkpoint = torch.load('conditional_vae_mnist.pth', map_location='cpu')
    model = ConditionalVAE(latent_dim=20, num_classes=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def generate_digits(model, digit, num_samples=5):
    model.eval()
    with torch.no_grad():
        labels = torch.full((num_samples,), digit, dtype=torch.long)
        z = torch.randn(num_samples, 20)
        generated = model.decode(z, labels)
        generated = generated.view(num_samples, 1, 28, 28)
        generated = (generated + 1) / 2
    return generated

st.title("🔢 Handwritten Digit Generator")
st.write("Generate 5 unique handwritten digits using AI")

model = load_model()
digit = st.selectbox("Choose digit (0-9):", range(10))

if st.button("Generate 5 Images"):
    with st.spinner("Generating..."):
        samples = generate_digits(model, digit, num_samples=5)
        
        st.success(f"Generated 5 images of digit {digit}!")
        cols = st.columns(5)
        
        for i in range(5):
            with cols[i]:
                img = samples[i].squeeze().numpy()
                img = (img * 255).astype('uint8')
                st.image(img, caption=f"Sample {i+1}")
