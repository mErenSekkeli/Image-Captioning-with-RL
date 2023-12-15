import json
import os
import cv2
import tarfile
import torch
import numpy as np

def load_json_data(json_file, start_index, end_index):

    with open(json_file, "r") as f:
        data = json.load(f)
    
    data = data["image"]

    grouped_data = []
    for i in range(start_index * 4 - 4, end_index * 4 + 4, 4):  # Her dört adımda bir ilerle
        group = data[i:i + 4]  # Her grupta dört eleman al
        grouped_data.append(group)

    return grouped_data

def load_image_filenames(json_file, start_index, end_index):
    data = load_json_data(json_file, start_index, end_index)
    filenames = []

    for group in data:
        for item in group:
            filename = item["filename"].split("_")[0]
            if filename not in filenames:
                filenames.append(filename)

    return filenames

def load_images(image_type = "before", numpy=False):

    if image_type == "before":
        data_folder = "data/im1"
    elif image_type == "after":
        data_folder = "data/im2"

    images = []
    for filename in os.listdir(data_folder):
        img = cv2.imread(os.path.join(data_folder, filename))
        if img is not None:
            images.append(img)
        else:
            print(f"Belirtilen dosya bulunamadı: {filename}")
    
    if numpy:
        return np.array(images)
    else:
        return images
    
def load_sentences(json_file, start_index, end_index):
    data = load_json_data(json_file, start_index, end_index)
    sentences = []

    for group in data:
        for item in group:
            sentence1 = item["sentence1"]
            sentence2 = item["sentence2"]
            sentence3 = item["sentence3"]

            sentences.append(sentence1)
            sentences.append(sentence2)
            sentences.append(sentence3)            
    return np.array(sentences)

def image_load(file_name="00003"):
    # Dosyanın bulunduğu klasör yolu
    data_folder = "data"  # Verilerin bulunduğu klasörün adını güncelleyin
    im1_folder = os.path.join(data_folder, "im1")
    im2_folder = os.path.join(data_folder, "im2")

    # İlk resmi oku
    im1 = cv2.imread(os.path.join(im1_folder, f"{file_name}.png"))
    im2 = cv2.imread(os.path.join(im2_folder, f"{file_name}.png"))

    if im1 is None:
        return None  # Resim okunamazsa None döndür

    # Resmin boyutlarını al
    height, width, _ = im1.shape

    # Her bir resmi 4 eşit parçaya böl
    quarter_height = height // 4
    quarter_width = width // 4
    im1_parts = []
    im2_parts = []

    for i in range(4):
        for j in range(4):
            im1_part = im1[i * quarter_height : (i + 1) * quarter_height, j * quarter_width : (j + 1) * quarter_width]
            im2_part = im2[i * quarter_height : (i + 1) * quarter_height, j * quarter_width : (j + 1) * quarter_width]

            if im2_part is None:
                return None  # Resim okunamazsa None döndür

            im1_parts.append(im1_part)
            im2_parts.append(im2_part)

    return [im1_parts, im2_parts]

def load_extracted_features(file_name):
    # load the npy file and return the features
    features = np.load(f"data/features/{file_name}.npy")
    return features


load_sentences("data/result.json", 1000, 1519)

# Test
"""if __name__ == "__main__":
    loaded_result_json_data = load_json_data("data/result.json", 1000, 1520)
    #write if not exists
    if not os.path.exists("data/loaded_result.json"):
        with open("data/loaded_result.json", "w") as f:
            json.dump(loaded_result_json_data, f)
    else:
        print("File already exists!")

    images = image_load("00003")

    if images:
        im1_parts, im2_parts = images
        for i in range(4):
            if im1_parts[i] is not None:
                print(f"İlk Resim {i+1} (im1) Boyutu:", im1_parts[i].shape)
            else:
                print(f"İlk Resim {i+1} (im1) okunamadı.")
            
            if im2_parts[i] is not None:
                print(f"İkinci Resim {i+1} (im2) Boyutu:", im2_parts[i].shape)
            else:
                print(f"İkinci Resim {i+1} (im2) okunamadı.")
    else:
        print("Belirtilen dosya bulunamadı.")"""
    