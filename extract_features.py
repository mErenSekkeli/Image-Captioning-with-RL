from load_data import load_images
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()


def extract_features(image_array, batch_size):
    # Load the pre-trained ResNet101 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet101(pretrained=True).to(device)
    # Remove the last fully-connected layer
    modules = torch.nn.Sequential(*list(model.children())[:-1])
    model = modules.to(device)

    # Define transformations
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply transformations and create a DataLoader
    tensor_images = torch.stack([preprocess(image) for image in image_array])
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
    parser.add_argument("--json_file", type=str, required=False, default="data/result.json", help="JSON file path")
    parser.add_argument("--start_index", type=int, required=False, default=1000, help="Start index")
    parser.add_argument("--end_index", type=int, required=False, default=1520, help="End index")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction")
    return parser.parse_args()

def main():
    if os.path.exists("data/features/im1.npy") and os.path.exists("data/features/im2.npy"):
        raise Exception("Features already extracted")
    args = parse_args()
    print(args)

    im1_data = load_images(image_type="before")
    
    im1_features = extract_features(im1_data, args.batch_size)
    print("im1_features shape:", im1_features.shape)
    save_features(im1_features, "im1")
    del im1_features
    im2_data = load_images(image_type="after")
    im2_features = extract_features(im2_data, args.batch_size)
    print("im2_features shape:", im2_features.shape)
    save_features(im2_features, "im2")

if __name__ == "__main__":
    main()
