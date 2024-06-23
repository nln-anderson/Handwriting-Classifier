import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.utils import make_grid
from torch import nn
import torch.nn.functional as F
from collections import Counter
from PIL import Image, ImageOps
import random
from torchvision import transforms

# For reproducibility
torch.manual_seed(42)

# Custom Transformation for Zoom Out Only
import torchvision.transforms.functional as TF
import random
from PIL import Image

class RandomZoomOut:
    def __init__(self, size, scale=(0.6, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img):
        # Randomly scale the image
        scale_factor = random.uniform(*self.scale)
        new_size = int(self.size * scale_factor)
        
        # Resize the image using LANCZOS resampling
        img = img.resize((new_size, new_size), Image.Resampling.LANCZOS)
        
        # Create a new white background image
        new_img = Image.new("L", (self.size, self.size), 255)
        
        # Paste the resized image onto the center of the new image
        paste_x = (self.size - new_size) // 2
        paste_y = (self.size - new_size) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img


# Load image folder with updated transformations
class ConditionalGaussianBlur:
    def __init__(self, probability=0.5, kernel_size=(3, 3), sigma=(0.05, 1.0)):
        self.probability = probability
        self.blur_transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, img):
        if random.random() < self.probability:
            return self.blur_transform(img)
        return img

# Load image folder with updated transformations
transforms = transforms.Compose([
    RandomZoomOut(45, scale=(0.3, 1.0)),  # Randomly zoom out and pad
    ConditionalGaussianBlur(probability=0.5, kernel_size=(3, 3), sigma=(0.05, 1)),  # Apply Gaussian blur conditionally
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])

dataset = ImageFolder("C:/Users/nolio/OneDrive/Desktop/Datasets/archive2/data/extracted_images", transform=transforms)

print("Class distribution in the dataset:", dict(Counter(dataset.targets)))

# Split the data
test_pct = 0.3
test_size = int(len(dataset) * test_pct)
dataset_size = len(dataset) - test_size

val_pct = 0.1
val_size = int(dataset_size * val_pct)
train_size = dataset_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
print(f"Dataset sizes - Train: {len(train_ds)}, Validation: {len(val_ds)}, Test: {len(test_ds)}")

batch_size = 64

# Calculate class weights for weighted sampling
class_counts = Counter(train_ds.dataset.targets[i] for i in train_ds.indices)
total_samples = len(train_ds)
weights = torch.zeros(len(class_counts))
for class_idx, count in class_counts.items():
    weights[class_idx] = total_samples / (len(class_counts) * count)

# Create a list of weights for each sample in the training set
sample_weights = [weights[train_ds.dataset.targets[i]] for i in train_ds.indices]

# Data loaders with WeightedRandomSampler
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(train_ds))

# Data loaders
train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
val_dl = DataLoader(val_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size)

# Look at a batch
def show_batch(dl):
    for img, lb in dl:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(img.cpu(), nrow=16).permute(1, 2, 0))
        break

show_batch(train_dl)
plt.show()

raise ValueError
# Network Building
class MathNet(nn.Module):
    def __init__(self):
        super(MathNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, padding=2)  # 20 filters of size 5x5 with padding
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5, padding=2)  # 40 filters of size 5x5 with padding
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling with a 2x2 kernel and stride 2
        
        # Define the new fully connected layers
        self.fc1 = nn.Linear(40 * 11 * 11, 400)  # First fully connected layer with 400 neurons
        self.fc2 = nn.Linear(400, 200)  # Second fully connected layer with 200 neurons
        self.fc3 = nn.Linear(200, 80)  # Output layer with 80 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply first convolutional layer, ReLU, and max pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply second convolutional layer, ReLU, and max pooling
        x = x.view(-1, 40 * 11 * 11)  # Flatten the output for fully connected layers
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first fully connected layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second fully connected layer
        x = self.fc3(x)  # Output layer
        return x


# Check if GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MathNet().to(device)

# Defining loss and parameters
learning_rate = 0.001
epochs = 5
batch_size = 64

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    running_loss = 0.0
    model.train()
    for i, data in enumerate(train_dl, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    # Evaluate on validation set
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Validation Loss: {val_loss / len(val_dl):.3f}, Accuracy: {100 * correct / total:.2f}%')

print('Finished Training')

# Save the model
PATH = './math_net_with_weights_6.pth'
torch.save(model.state_dict(), PATH)

# Testing loop
model.load_state_dict(torch.load(PATH))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in test_dl:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')

# Class-wise accuracy
correct_pred = {classname: 0 for classname in dataset.classes}
total_pred = {classname: 0 for classname in dataset.classes}

with torch.no_grad():
    for data in test_dl:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[dataset.classes[label]] += 1
            total_pred[dataset.classes[label]] += 1

# Print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class {classname} is {accuracy:.2f} %')









# Testing on Individual Images
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch

# Load the image
image_path = "C:/Users/nolio/OneDrive/Desktop/Datasets/archive2/data/extracted_images/theta/exp1222.jpg"
image = Image.open(image_path)

# Define transformations
# Load image folder
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# Apply transformations
input_tensor = transforms(image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# Move the input to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_batch = input_batch.to(device)

# Set your model to evaluation mode
model.eval()

# Perform inference
with torch.no_grad():
    output = model(input_batch)

# Get class probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)


# Get predicted class
predicted_class = torch.argmax(probabilities).item()
confidence = torch.max(probabilities).item()

# Map the predicted index to the class name
predicted_class_name = dataset.classes[predicted_class]

# Print the predicted class and confidence
print("Predicted class:", predicted_class_name)
print("Confidence:", confidence)






# UI Stuff
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageDraw, ImageOps
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
# Define the size of the canvas and the image
canvas_size = 200  # Canvas size in pixels
img_size = 45  # Size of the image to match the model input

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MathNet()
model.load_state_dict(torch.load('math_net_with_weights_6.pth'))
model.to(device)
model.eval()

# Define transformations
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((45, 45)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create the main window
root = tk.Tk()
root.title("Symbol Recognition")

# Create a canvas for drawing
canvas = Canvas(root, width=canvas_size, height=canvas_size, bg='white')
canvas.pack()

# Create an empty image and a drawing object
image1 = Image.new('RGB', (canvas_size, canvas_size), 'white')
draw = ImageDraw.Draw(image1)

# Function to draw on the canvas
def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=1)  # Pen stroke width reduced
    draw.line([x1, y1, x2, y2], fill='black', width=2)  # Pen stroke width reduced

canvas.bind("<B1-Motion>", paint)

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill='white')

# Function to predict the drawn symbol
def predict():
    # Convert the image to grayscale
    img = image1.convert('L')
    
    # Apply a binary threshold to make the penstrokes strictly black
    img = np.array(img)
    threshold_value = 128
    img = (img > threshold_value) * 255  # Binarize the image
    img = Image.fromarray(img.astype(np.uint8))

    # Apply transformations
    img = preprocess(img)
    img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Visualize the preprocessed image
    plt.imshow(img.cpu().squeeze(), cmap='gray')
    plt.show()

    # Perform inference
    with torch.no_grad():
        output = model(img)
        predicted_class_idx = torch.argmax(output).item()
        predicted_class_name = dataset.classes[predicted_class_idx]

    # Display the result
    result_label.config(text=f"Predicted: {predicted_class_name}")



# Create buttons for clearing and predicting
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

# Create a label to display the prediction result
result_label = tk.Label(root, text="Draw a symbol and click Predict")
result_label.pack()

# Run the Tkinter main loop
root.mainloop()
