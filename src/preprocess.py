import os
from PIL import Image
from torchvision import transforms

def preprocess_image(image_path, save_path=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor(),         # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    img = Image.open(image_path)
    img_tensor = transform(img)
    
    if save_path:
        img.save(save_path)  # Save preprocessed image if needed
    return img_tensor

def preprocess_all_images(input_dir):
    data = []
    labels = []
    for label, folder in enumerate(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                data.append(preprocess_image(file_path))
                labels.append(label)
    return data, labels
