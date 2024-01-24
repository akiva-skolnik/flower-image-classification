import argparse
import json
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def train(data_directory: str, save_directory: str, architecture: str,
          learning_rate: float, hidden_units: tuple[int], epochs: int, gpu: bool):
    """train a new network on a dataset and save the model as a checkpoint.
    Prints out training loss, validation loss, and validation accuracy as the network trains"""

    device = get_device(gpu)
    train_loader, valid_loader, test_loader = get_loaders(data_directory, device)
    model = load_model(architecture)
    set_classifier(model, hidden_units, len(train_loader.dataset.classes))
    model.class_to_idx = train_loader.dataset.class_to_idx
    criterion, optimizer = get_loss_and_optimizer(model, learning_rate)

    logging.info("Start training")
    train_losses, valid_losses, accuracies = train_model(
        model, criterion, optimizer, train_loader, valid_loader, epochs, device)
    logging.info("Done training")

    save_model(model, optimizer, architecture, save_directory, learning_rate)

    plot_losses_and_accuracies(train_losses, valid_losses, accuracies)

    logging.info("Calculating test loss and accuracy")
    test_loss, accuracy = get_loss_and_accuracy(model, test_loader, criterion, device)
    logging.info(f"Test loss: {test_loss:.3f}.. "
                 f"Test accuracy: {accuracy:.2%}")


def save_model(model: torchvision.models, optimizer, architecture: str, save_directory: str, learning_rate: float,
               file_name: str = f'checkpoint_{datetime.now().strftime("%Y%m%d%H%M%S")}.pth'):
    """save the model"""
    checkpoint = {
        'architecture': architecture,
        # Exclude the output layer
        'hidden_layers': [each.out_features for each in model.classifier if hasattr(each, 'out_features')][:-1],
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'learning_rate': learning_rate,
    }
    logging.info(f"Saving model to {os.path.join(save_directory, file_name)}")
    torch.save(checkpoint, os.path.join(save_directory, file_name))


def load_checkpoint(filepath: str, device: torch.device) -> torchvision.models:
    """load the model"""
    checkpoint = torch.load(filepath, map_location=device)
    model = load_model(checkpoint['architecture'])
    set_classifier(model, checkpoint['hidden_layers'], len(checkpoint['class_to_idx']))
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    learning_rate = checkpoint['learning_rate']
    criterion, optimizer = get_loss_and_optimizer(model, learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model


def load_model(architecture: str) -> torchvision.models:
    """load the model"""
    all_weights = {k.lower(): v for k, v in torchvision.models.__dict__.items() if k.endswith("_Weights")}
    try:
        weights = all_weights[architecture.lower() + "_weights"].DEFAULT
    except KeyError:
        logging.warning(f"Architecture {architecture} not found, using default weights")
        weights = None
    model = getattr(torchvision.models, architecture)(weights=weights)
    for param in model.parameters():
        param.requires_grad = False  # freeze parameters
    logging.info("Model loaded: " + str(model))
    return model


def get_in_features(model: torchvision.models) -> int:
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, torch.nn.Sequential):
            classifier = model.classifier
        else:
            classifier = torch.nn.Sequential(model.classifier)
    elif hasattr(model, 'fc'):
        if isinstance(model.fc, torch.nn.Sequential):
            classifier = model.fc
        else:
            classifier = torch.nn.Sequential(model.fc)
    else:
        raise ValueError("Could not find classifier")

    in_features = None
    for layer in classifier:
        if hasattr(layer, 'in_features'):  # skip layers without in_features like ReLU, Dropout, etc.
            in_features = layer.in_features
            break
    if in_features is None:
        raise ValueError("Could not find in_features")
    logging.info(f"in_features: {in_features}")
    return in_features


def set_classifier(model: torchvision.models, hidden_units: tuple[int], out_features: int):
    """build the model"""
    in_features = get_in_features(model)

    layers = [
        torch.nn.Linear(in_features, hidden_units[0]),
        torch.nn.ReLU(),
    ]
    for i in range(1, len(hidden_units)):
        layers.append(torch.nn.Linear(hidden_units[i - 1], hidden_units[i]))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(hidden_units[-1], out_features))
    layers.append(torch.nn.LogSoftmax(dim=1))

    classifier = torch.nn.Sequential(*layers)
    model.classifier = classifier
    logging.info("Classifier set to: " + str(classifier))


def plot_losses_and_accuracies(train_losses: list, valid_losses: list, accuracies: list):
    """plot the losses and accuracies"""
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2, nrows=1)
    ax1.plot(train_losses, label='Training loss')
    ax1.plot(valid_losses, label='Validation loss')
    ax1.legend(frameon=False)
    ax2.plot(accuracies, label='Validation accuracy')
    plt.show()


def training_step(model: torchvision.models, inputs: torch.Tensor, labels: torch.Tensor,
                  criterion: torch.nn, optimizer: torch.optim, device: torch.device) -> float:
    """perform a training step"""
    inputs, labels = inputs.to(device), labels.to(device)

    logps = model.forward(inputs)  # forward pass
    loss = criterion(logps, labels)  # calculating loss

    optimizer.zero_grad()  # zeroing out accumulated gradients
    loss.backward()  # backpropagation
    optimizer.step()  # optimization step
    training_loss = loss.item()
    logging.debug(f"Training loss: {training_loss:.3f}")
    return training_loss


def train_model(model: torchvision.models, criterion: torch.nn, optimizer: torch.optim,
                train_loader: torch.utils.data.DataLoader, valid_loader: torch.utils.data.DataLoader,
                epochs: int, device: torch.device) -> (list, list, list):
    """train the model"""
    train_losses = []
    valid_losses = []
    accuracies = []
    model.to(device)
    for epoch in range(1, epochs + 1):
        try:
            running_loss = 0
            model.train()
            for inputs, labels in train_loader:
                running_loss += training_step(model, inputs, labels, criterion, optimizer, device)

            valid_loss, accuracy = get_loss_and_accuracy(model, valid_loader, criterion, device)
            running_loss /= len(train_loader)
            logging.info(f"Epoch {epoch}/{epochs}.. "
                         f"Train loss: {running_loss:.3f}.. "
                         f"Val loss: {valid_loss :.3f}.. "
                         f"Val accuracy: {accuracy:.3f}")

            train_losses.append(running_loss)
            valid_losses.append(valid_loss)
            accuracies.append(accuracy)
        except KeyboardInterrupt:
            print("Interrupted by user")
            break
    logging.debug(f"train_losses: {train_losses}; valid_losses: {valid_losses}; accuracies: {accuracies}")
    return train_losses, valid_losses, accuracies


def get_loss_and_accuracy(model: torchvision.models, data_loader: torch.utils.data.DataLoader,
                          criterion: torch.nn, device: torch.device) -> (float, float):
    """return the loss and accuracy"""
    model.eval()
    loss = 0
    accuracy = 0
    with torch.inference_mode():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss += criterion(logps, labels).item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = (top_class == labels.view(*top_class.shape))
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return loss / len(data_loader), accuracy / len(data_loader)


def get_loss_and_optimizer(model: torchvision.models, lr: float) -> (torch.nn, torch.optim):
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    return criterion, optimizer


def get_cat_to_name(file_path: str = 'cat_to_name.json') -> dict:
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def get_device(gpu: bool) -> torch.device:
    """return the device to use"""
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logging.info(f"Using {'GPU' if str(device) == 'cuda' else 'CPU'}")
    return device


def get_loaders(data_directory: str, device: torch.device, batch_size: int = 64):
    """load the data from the data directory and return the data _loaders"""
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'

    train_transforms, valid_transforms, test_transforms = get_transforms()
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=test_transforms)

    num_workers = os.cpu_count() if str(device) == "cuda" else 0
    pin_memory = str(device) == "cuda"
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                               num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    return train_loader, valid_loader, test_loader


def get_transforms(resize: int = 255, crop: int = 224, rotate: int = 15) -> (
        torchvision.transforms, torchvision.transforms, torchvision.transforms):
    """return the transforms for the training, validation and testing data"""
    means, stds = get_mean_std()
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(rotate),
        torchvision.transforms.RandomResizedCrop(crop),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(means, stds)
    ])
    valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.CenterCrop(crop),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(means, stds)
    ])
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.CenterCrop(crop),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(means, stds)
    ])
    return train_transforms, valid_transforms, test_transforms


def get_mean_std() -> (np.array, np.array):
    """return the mean and standard deviation of the data"""
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    return means, stds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Train a new network on a data set')
    parser.add_argument('data_directory', type=str, help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default=os.getcwd(), help='Path to save directory')
    parser.add_argument('--arch', type=str, default='vgg16', help='Architecture of the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate of the model')
    parser.add_argument('--hidden_units', type=int, nargs='+', default=[256, 128],
                        help='Number of hidden layers and units of the model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs of the model')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    try:
        train(args.data_directory, args.save_dir, args.arch, args.learning_rate,
              args.hidden_units, args.epochs, args.gpu)
        print("Done train.py")
    except Exception as e:
        print('Error: training failed with error ' + str(e))
        raise  # we have nothing better to do here.
