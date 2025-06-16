from torchvision.io import read_image
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt

from blur_generator import generate_synthetic_burst

# Load image from relative path
img_path = "../photos/clean/27800019.jpeg"
img = read_image(img_path).float() / 255.0  # Normalize to [0, 1]

# Generate burst
burst = generate_synthetic_burst(img, num_variants=5)

# Visualize
for i, variant in enumerate(burst):
    plt.subplot(1, 5, i + 1)
    plt.imshow(variant.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Burst {i + 1}")
    plt.axis('off')
plt.show()
