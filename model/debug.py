import glob, os

# Absolute path to your folder of clean images
clean_dir = r"C:\Users\spenc\OneDrive\Documents\Deblurring\photos\clean"
# Or: clean_dir = "C:/Users/spenc/OneDrive/Documents/Deblurring/photos/clean"

# Gather .png and .jpg
paths = sorted(
    glob.glob(os.path.join(clean_dir, "*.png")) +
    glob.glob(os.path.join(clean_dir, "*.jpg"))
)

print(f"Loading clean images from: {clean_dir}")
print(f"Found {len(paths)} files:")
for p in paths:
    print("  ", p)
