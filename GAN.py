import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Discriminator import Discriminator
from Generator import Generator
import joblib

latent_dim = 100
lr_G = 0.0002
lr_D = 0.0001
batch_size = 128
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
generator = Generator(latent_dim=latent_dim).to(device)
discriminator = Discriminator().to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))
criterion = nn.BCELoss()
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.to(device)
        batch_size = real_imgs.shape[0]
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs), torch.ones(batch_size, 1).to(device))
        fake_loss = criterion(discriminator(fake_imgs.detach()), torch.zeros(batch_size, 1).to(device))
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(fake_imgs), torch.ones(batch_size, 1).to(device))
        g_loss.backward()
        optimizer_G.step()
        
    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

print("Training Complete!")
joblib.dump(generator.state_dict(), "generator.pkl")

