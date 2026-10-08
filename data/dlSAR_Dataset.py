import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "bakhtiyar2222/deep-sar-oil-spill-segmentation-refined"
)

print("Path to downloaded dataset files:", path)