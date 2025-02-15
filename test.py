import joblib
import torch
from Generator import Generator
from matplotlib import pyplot as plt

latent_dim = 100
generator = Generator(latent_dim=latent_dim)
generator.load_state_dict(joblib.load("generator.pkl"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
generator.eval()
print("Generator model loaded successfully!")
z = torch.randn(1, latent_dim).to(device)
with torch.no_grad():
    fake_img = generator(z).cpu().squeeze()
plt.imshow(fake_img, cmap="gray")
plt.show()