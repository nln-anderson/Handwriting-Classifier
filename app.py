# This is the app for the handwriting classifier. It only contains the network and GUI elements. To see training
# info, please look at math_6.py.

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

model = MathNet()
classes = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
           '=', 'A', 'C', 'Delta', 'G', 'H', 'M', 'N', 'R', 'S', 'T', 'X', '[', ']', 'alpha', 
           'b', 'beta', 'cos', 'd', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 'gamma', 
           'geq', 'gt', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots', 'leq', 'lim', 'log', 
           'lt', 'mu', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'prime', 'q', 'rightarrow', 'sigma', 'sin', 'sqrt', 
           'sum', 'tan', 'theta', 'u', 'v', 'w', 'y', 'z', '{', '}']

# Using the trained model weights
PATH = './math_net_with_weights_6.pth'
model.load_state_dict(torch.load(PATH))

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
    # plt.imshow(img.cpu().squeeze(), cmap='gray')
    # plt.show()

    # Perform inference
    with torch.no_grad():
        output = model(img)
        predicted_class_idx = torch.argmax(output).item()
        predicted_class_name = classes[predicted_class_idx]

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

