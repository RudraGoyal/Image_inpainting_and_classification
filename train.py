import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

# Define VAE architecture
class VAE(nn.Module):
    def __init__(self, image_size=256, latent_dim=100):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(512 * (image_size // 16) * (image_size // 16), latent_dim)
        self.fc_logvar = nn.Linear(512 * (image_size // 16) * (image_size // 16), latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * (image_size // 16) * (image_size // 16)),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(z.size(0), 512, (image_size // 16), (image_size // 16))
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# Custom dataset class (replace this with your dataset loading code)

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # Load the corrupted and original images
        corrupted_img = Image.open(img_path).convert('RGB')
        original_img = Image.open(img_path.replace('_corrupted', '')).convert('RGB')

        # Apply transformations
        if self.transform:
            corrupted_img = self.transform(corrupted_img)
            original_img = self.transform(original_img)

        return corrupted_img, original_img

# Example usage:
# Define transformations (adjust as needed)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Specify the path to your dataset
dataset_root = './datasets/train/'

# Create the dataset
image_dataset = ImageDataset(root_dir=dataset_root, transform=transform)

# Create the dataloader
batch_size = 64
dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Set hyperparameters
image_size = 256
latent_dim = 100
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Create VAE model
vae = VAE(image_size=image_size, latent_dim=latent_dim)

# Create dataset and dataloader (replace with your dataset loading code)
# Assuming you have corrupted_images and original_images lists
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = CustomDataset(corrupted_images, original_images, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# Optimizer
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (corrupted_batch, original_batch) in enumerate(dataloader):
        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar = vae(corrupted_batch)

        # Compute loss
        loss = loss_function(recon_batch, original_batch, mu, logvar)

        # Backward pass
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(vae.state_dict(), 'vae_inpainting_model.pt')
