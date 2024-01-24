import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image

from train import get_device, get_mean_std, load_checkpoint


def process_image(image: Image, means: np.array, stds: np.array) -> np.ndarray:
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    """
    image.thumbnail((256, 256), Image.LANCZOS)  # ANTIALIAS to keep the aspect ratio
    image = image.crop((16, 16, 240, 240))  # Crop out the center 224x224 portion of the image
    image = np.array(image) / 255  # Convert color channels to floats 0-1
    image = (image - means) / stds  # Normalize
    image = image.transpose((2, 0, 1))  # Reorder dimensions
    return image


def plot_probabilities(image_path: str, top_p: list[float], top_class: list[str], title: str) -> None:
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=1, nrows=2)
    # process_image(Image.open(image_path)).transpose((1, 2, 0))
    ax1.imshow(Image.open(image_path))
    ax1.set_title(title)
    ax1.axis('off')
    ax2.barh(np.arange(len(top_class)), top_p)  # horizontal bar plot
    ax2.set_yticks(np.arange(len(top_class)))
    ax2.set_yticklabels(top_class)
    ax2.set_title('Class Probability')

    plt.tight_layout()
    plt.show()


def predict(image_path: str, model: torchvision.models, topk: int, device: torch.device) -> (list[float], list[str]):
    """Predict the class (or classes) of an image using a trained deep learning model.
    """
    means, stds = get_mean_std()
    image = Image.open(image_path)
    image = process_image(image, means, stds)
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)  # Add batch dimension
    model.eval()
    with torch.inference_mode():
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p = top_p.numpy()[0]
        top_class = top_class.numpy()[0]
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_class = [idx_to_class[each] for each in top_class]
    return top_p, top_class


def load_cat_to_name(category_names_path: str) -> dict[str, str]:
    """Load category names from JSON file."""
    if not os.path.exists(category_names_path):
        raise FileNotFoundError(f"File not found: {category_names_path}")
    else:
        with open(category_names_path, 'r') as f:
            cat_to_name = json.load(f)
    return cat_to_name


def main(image_path: str, checkpoint_path: str, top_k: int, category_names_path: str, gpu: bool):
    """Predict flower name from an image with predict.py along with the probability of that name.
    That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

    Basic usage: python predict.py /path/to/image checkpoint
    Options:
        * Return top K most likely classes: python predict.py input checkpoint --top_k 3
        * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
        * Use GPU for inference: python predict.py input checkpoint --gpu
    """
    device = get_device(gpu)

    # Load model
    model = load_checkpoint(checkpoint_path, device)

    # Predict
    probs, classes = predict(image_path, model, top_k, device)

    # Load category names
    cat_to_name = load_cat_to_name(category_names_path)

    # Print results
    for prob, class_name in zip(probs, classes):
        print(f"{cat_to_name[class_name]}: {prob}")

    # Plot results
    top_class = [cat_to_name[class_name] for class_name in classes]
    real_class = cat_to_name[image_path.split('\\' if '\\' in image_path else "/")[-2]]
    plot_probabilities(image_path, probs, top_class, title=real_class)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Predict flower name from an image along with the probability of that name.")
    parser.add_argument("image_path", help="Path to image")
    parser.add_argument("checkpoint_path", help="Checkpoint file")
    parser.add_argument("--top_k", help="Return top K most likely classes", type=int, default=1)
    parser.add_argument("--category_names", help="Use a mapping of categories to real names",
                        default="cat_to_name.json")
    parser.add_argument("--gpu", help="Use GPU for inference", action="store_true")
    args = parser.parse_args()

    # Run
    main(args.image_path, args.checkpoint_path, args.top_k, args.category_names, args.gpu)
