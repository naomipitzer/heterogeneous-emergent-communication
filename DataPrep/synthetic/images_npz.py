"""
Synthetic Shape Image Generation, VGG16 Fine-tuning / Embedding Extraction

Generates synthetic shape images, fine-tunes VGG16 (if needed), extracts embeddings,
applies optional preprocessing, and saves to .npz file.

Usage:
    python synthetic_image_embeddings.py
        --instances_per_class=400
        --image_size=256
        --file_name=embeddings
        --preprocessing=none    # options: none, zeromean

Outputs:
    embeddings saved as:
        embeddings.npz         (no preprocessing)
        embeddings-zm.npz      (zero-mean preprocessing)

Arguments:
    --instances_per_class: Number of images per shape class (default: 400)
    --image_size: Width and height of generated square images (default: 256)
    --file_name: Base output filename (default: 'embeddings')
    --preprocessing: Embedding preprocessing (none, zeromean) (default: none)
    --save_vgg16: Whether to save the fine-tuned VGG16 model (default:False)
"""

import os
import random
import math
import argparse
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, models
from tqdm import tqdm

# --- Shape drawing functions ---

def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def draw_shape(draw, shape, bounds, color):
    x0, y0, x1, y1 = bounds
    if shape == "circle":
        draw.ellipse(bounds, fill=color, outline=color)
    elif shape == "triangle":
        draw.polygon([(x0, y1), ((x0 + x1)/2, y0), (x1, y1)], fill=color, outline=color)
    elif shape == "square":
        draw.rectangle(bounds, fill=color, outline=color)
    elif shape == "hexagon":
        width = x1 - x0
        height = y1 - y0
        draw.polygon([
            (x0 + width*0.25, y0),
            (x0 + width*0.75, y0),
            (x1, y0 + height*0.5),
            (x0 + width*0.75, y1),
            (x0 + width*0.25, y1),
            (x0, y0 + height*0.5)
        ], fill=color, outline=color)
    elif shape == "star":
        width = x1 - x0
        height = y1 - y0
        cx, cy = x0 + width/2, y0 + height/2
        r = min(width, height)/2
        points = []
        for i in range(10):
            angle = i * math.pi / 5
            radius = r if i % 2 == 0 else r / 2
            points.append((cx + radius * math.cos(angle), cy - radius * math.sin(angle)))
        draw.polygon(points, fill=color, outline=color)
    elif shape == "heart":
        width = x1 - x0
        height = y1 - y0
        cx, cy = x0 + width/2, y0 + height/3
        top_curve_radius = width*0.25
        draw.polygon([
            (cx, y1),  # bottom
            (x0, cy),
            (cx, y0 + height*0.1),
            (x1, cy)
        ], fill=color, outline=color)
        draw.ellipse([x0, y0, x0 + 2*top_curve_radius, y0 + 2*top_curve_radius], fill=color, outline=color)
        draw.ellipse([x1 - 2*top_curve_radius, y0, x1, y0 + 2*top_curve_radius], fill=color, outline=color)

def rotate_image(img, angle):
    return img.rotate(angle, expand=True, fillcolor="white")

# --- Synthetic Image Dataset class (in memory) ---

class SyntheticShapesDataset(Dataset):
    def __init__(self, shapes, instances_per_class, image_size):
        self.shapes = shapes
        self.instances_per_class = instances_per_class
        self.image_size = image_size
        self.data = []
        self.labels = []
        self._generate_data()

    def _generate_data(self):
        for label, shape in enumerate(self.shapes):
            for _ in range(self.instances_per_class):
                img = Image.new("RGB", (self.image_size, self.image_size), "white")
                draw = ImageDraw.Draw(img)
                size = random.randint(self.image_size//6, self.image_size//2)
                x0 = random.randint(0, self.image_size - size)
                y0 = random.randint(0, self.image_size - size)
                bounds = (x0, y0, x0 + size, y0 + size)
                color = random_color()
                draw_shape(draw, shape, bounds, color)
                angle = random.randint(0, 360)
                img = rotate_image(img, angle)
                self.data.append(img)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Main ---

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shapes = ["circle", "triangle", "square", "hexagon", "star", "heart"]
    print(f"Generating synthetic dataset: {len(shapes)} shapes, {args.instances_per_class} instances each")

    dataset = SyntheticShapesDataset(shapes, args.instances_per_class, args.image_size)

    # Split 80/20 train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    # Data loaders with VGG transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    class TransformedDataset(Dataset):
        def __init__(self, base_ds, transform):
            self.base_ds = base_ds
            self.transform = transform
        def __len__(self):
            return len(self.base_ds)
        def __getitem__(self, idx):
            img, label = self.base_ds[idx]
            img_t = self.transform(img)
            return img_t, label

    train_ds = TransformedDataset(train_ds, transform)
    test_ds = TransformedDataset(test_ds, transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Load or fine-tune model
    model_path = "vgg16-finetuned.pth"
    vgg_model = models.vgg16(pretrained=True)

    # Freeze all layers except classifier last layer
    for param in vgg_model.parameters():
        param.requires_grad = False
    vgg_model.classifier[6] = nn.Linear(4096, 128)

    vgg_model.to(device)

    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}")
        vgg_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Fine-tuning VGG16 model on synthetic shapes dataset")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(vgg_model.classifier.parameters(), lr=1e-4)

        num_epochs = 10
        for epoch in range(num_epochs):
            vgg_model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = vgg_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)

            # Evaluate train and test accuracy
            train_acc = evaluate_accuracy(vgg_model, train_loader, device)
            test_acc = evaluate_accuracy(vgg_model, test_loader, device)

            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} Train Acc: {train_acc:.2f}% Test Acc: {test_acc:.2f}%")

        if(args.save_vgg16):
            torch.save(vgg_model.state_dict(), model_path)
            print(f"Saved fine-tuned model to {model_path}")

    # Switch to eval mode
    vgg_model.eval()

    # Extract embeddings for entire dataset (train+test)
    all_embeddings = []
    all_labels = []

    full_loader = DataLoader(TransformedDataset(dataset, transform), batch_size=32, shuffle=False)

    with torch.no_grad():
        for inputs, labels in tqdm(full_loader, desc="Extracting embeddings"):
            inputs = inputs.to(device)
            emb = vgg_model(inputs).cpu().numpy()
            all_embeddings.append(emb)
            all_labels.extend(labels.numpy())

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)

    # Preprocessing
    if args.preprocessing == "zeromean":
        mean = np.mean(all_embeddings, axis=0)
        all_embeddings = all_embeddings - mean
        out_filename = f"{args.file_name}-zm.npz"
    else:
        out_filename = f"{args.file_name}.npz"

    np.savez(out_filename, embeddings=all_embeddings, labels=all_labels, label_map={s:i for i,s in enumerate(shapes)})
    print(f"Saved embeddings ({all_embeddings.shape}) to {out_filename}")

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Shape Image Embeddings Generator")
    parser.add_argument("--instances_per_class", type=int, default=400, help="Number of images per shape class")
    parser.add_argument("--image_size", type=int, default=256, help="Size of square images (pixels)")
    parser.add_argument("--file_name", type=str, default="vgg16_embeddings", help="Base output filename")
    parser.add_argument("--preprocessing", type=str, default="none", choices=["none", "zeromean"], help="Embedding preprocessing mode")
    parser.add_argument("--save_vgg16", type=bool, default=False, help="Whether to save the fine-tuned VGG16 model (default False)")
    args = parser.parse_args()

    main(args)
