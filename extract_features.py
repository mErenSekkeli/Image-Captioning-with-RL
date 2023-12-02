from load_data import image_load, load_json_data, load_image_filenames
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet101
import cv2
import numpy as np
import os
import tarfile
from PIL import Image
import argparse
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser()



def load_model():
    # ResNet101 modelini yükle
    model = resnet101(pretrained=True)
    model.eval()  # Modeli değerlendirme moduna al

    if torch.cuda.is_available():
        model = model.cuda()

    return model

def extract_features(model, images, transform):
    all_features = []
    
    im1_parts, im2_parts = images
    total_parts = len(im1_parts)

    for i in range(total_parts):
        im1_part = im1_parts[i]
        im2_part = im2_parts[i]

        im1_part = transform(Image.fromarray(im1_part))
        im2_part = transform(Image.fromarray(im2_part))

        if torch.cuda.is_available():
            im1_part = im1_part.cuda()
            im2_part = im2_part.cuda()

        im1_part = im1_part.unsqueeze(0)
        im2_part = im2_part.unsqueeze(0)

        with torch.no_grad():
            im1_features = model(im1_part)
            im2_features = model(im2_part)

        im1_features = im1_features.squeeze(0).cpu().numpy()
        im2_features = im2_features.squeeze(0).cpu().numpy()

        features = np.concatenate((im1_features, im2_features), axis=0)
        all_features.append(features)

    return all_features

def save_features(features, file_name):
    # Özellikleri kaydet
    if not os.path.exists("data/features"):
        os.makedirs("data/features")
    torch.save(features, os.path.join("data/features", f"{file_name}.pt"))

def delete_features():
    # Özellik klösörünü sil
    if os.path.exists("data/features"):
        shutil.rmtree("data/features")

def create_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction script")
    parser.add_argument("--json_file", type=str, required=False, default="data/result.json", help="JSON file path")
    parser.add_argument("--start_index", type=int, required=False, default=1000, help="Start index")
    parser.add_argument("--end_index", type=int, required=False, default=1520, help="End index")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction")
    return parser.parse_args()

def main():
    if os.path.exists("data/features.tar.gz"):
        raise Exception("Features already extracted")
    args = parse_args()
    print(args)
    model = load_model()

    file_names = load_image_filenames(args.json_file, args.start_index, args.end_index)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for name in tqdm(file_names, desc="Processing images", leave=False):
        images = image_load(name)
        if images:
            features = extract_features(model, images, transform)
            save_features(features, name)
        else:
            print(f"Belirtilen dosya bulunamadı: {name}")

    create_tarfile("data/features.tar.gz", "data/features")
    delete_features()

if __name__ == "__main__":
    main()
