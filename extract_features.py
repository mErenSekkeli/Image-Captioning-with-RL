from load_data import load_images
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()


def rotate_image(image, angle=90):
    return F.rotate(image, angle)

def extract_features(image_array, batch_size, augment_func=None):
    # Load the pre-trained ResNet101 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet101(pretrained=True).to(device)
    # Remove the last fully-connected layer
    modules = torch.nn.Sequential(*list(model.children())[:-1])
    model = modules.to(device)

    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor_images = []
    for image in image_array:
        pil_image = Image.fromarray(image)

        if augment_func:
            # Apply augmentation
            pil_image = augment_func(pil_image)

        processed_image = preprocess(pil_image)
        tensor_images.append(processed_image)

    tensor_images = torch.stack(tensor_images)
    dataset = TensorDataset(tensor_images)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Extract features
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Extracting features"):
            inputs = batch[0].to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())

    return np.concatenate(features, axis=0)

def save_features(image_features, name):
    # save features
    if not os.path.exists("data/features"):
        os.makedirs("data/features")    
    np.save(f"data/features/{name}.npy", image_features)


def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction script")
    parser.add_argument("--add_augment", action="store_true", help="Add augmented images", default=False)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    if args.add_augment:
        if os.path.exists("data/features/im1.npy") and os.path.exists("data/features/im2.npy") and os.path.exists("data/features/im1_90_degree.npy") and os.path.exists("data/features/im2_90_degree.npy") and os.path.exists("data/features/im1_180_degree.npy") and os.path.exists("data/features/im2_180_degree.npy") and os.path.exists("data/features/im1_270_degree.npy") and os.path.exists("data/features/im2_270_degree.npy"):
            raise Exception("Features already extracted")
    else:
        if os.path.exists("data/features/im1.npy") and os.path.exists("data/features/im2.npy"):
            raise Exception("Features already extracted")
    

    im1_data = load_images(image_type="before")

    if not os.path.exists("data/features/im1.npy"):
        im1_features = extract_features(im1_data, args.batch_size)
        print("im1_features shape:", im1_features.shape)
        save_features(im1_features, "im1")
        del im1_features

    if not os.path.exists("data/features/im1_90_degree.npy"):
        print("Extracting 90 degree rotated images for im1...")
        im1_90_degree = extract_features(im1_data, args.batch_size, rotate_image)
        print("im1_90_degree shape:", im1_90_degree.shape)
        save_features(im1_90_degree, "im1_90_degree")
        del im1_90_degree

    if not os.path.exists("data/features/im1_180_degree.npy"):
        print("Extracting 180 degree rotated images for im1...")
        im1_180_degree = extract_features(im1_data, args.batch_size, lambda x: rotate_image(x, 180))
        print("im1_180_degree shape:", im1_180_degree.shape)
        save_features(im1_180_degree, "im1_180_degree")
        del im1_180_degree

    if not os.path.exists("data/features/im1_270_degree.npy"):
        print("Extracting 270 degree rotated images for im1...")
        im1_270_degree = extract_features(im1_data, args.batch_size, lambda x: rotate_image(x, 270))
        print("im1_270_degree shape:", im1_270_degree.shape)
        save_features(im1_270_degree, "im1_270_degree")
        del im1_270_degree

    im2_data = load_images(image_type="after")
    if not os.path.exists("data/features/im2.npy"):
        im2_features = extract_features(im2_data, args.batch_size)
        print("im2_features shape:", im2_features.shape)
        save_features(im2_features, "im2")
        del im2_features


    if not os.path.exists("data/features/im2_90_degree.npy"):
        print("Extracting 90 degree rotated images for im2...")
        im2_90_degree = extract_features(im2_data, args.batch_size, rotate_image)
        print("im2_90_degree shape:", im2_90_degree.shape)
        save_features(im2_90_degree, "im2_90_degree")
        del im2_90_degree

    if not os.path.exists("data/features/im2_180_degree.npy"):
        print("Extracting 180 degree rotated images for im2...")
        im2_180_degree = extract_features(im2_data, args.batch_size, lambda x: rotate_image(x, 180))
        print("im2_180_degree shape:", im2_180_degree.shape)
        save_features(im2_180_degree, "im2_180_degree")
        del im2_180_degree

    if not os.path.exists("data/features/im2_270_degree.npy"):
        print("Extracting 270 degree rotated images for im2...")
        im2_270_degree = extract_features(im2_data, args.batch_size, lambda x: rotate_image(x, 270))
        print("im2_270_degree shape:", im2_270_degree.shape)
        save_features(im2_270_degree, "im2_270_degree")
        del im2_270_degree

if __name__ == "__main__":
    main()
