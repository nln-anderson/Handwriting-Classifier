# Application for converting handwriting to LaTeX code

import tkinter as tk
from tkinter import filedialog, Canvas
from torch import nn
import torch.nn.functional as F
import torch
from PIL import Image, ImageOps, ImageDraw
import numpy as np

# Network architecture
class MathNet(nn.Module):
    """
    Convolution neural network that was trained in other py file.
    """
    def __init__(self):
        super(MathNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(40 * 11 * 11, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 80)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 40 * 11 * 11)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Model:
    """
    Model for the application. Responsible for all backend operations
    """
    # Instance Variables
    network: MathNet

    # Methods
    def __init__(self) -> None:
        self.classes = [
            '!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'C', 'Delta', 'G',
            'H', 'M', 'N', 'R', 'S', 'T', 'X', '[', ']', 'alpha', 'b', 'beta', 'cos', 'd', 'div', 'e', 'exists', 'f',
            'forall', 'forward_slash', 'gamma', 'geq', 'gt', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots',
            'leq', 'lim', 'log', 'lt', 'mu', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'prime', 'q', 'rightarrow', 'sigma',
            'sin', 'sqrt', 'sum', 'tan', 'theta', 'u', 'v', 'w', 'y', 'z', '{', '}'
        ]
        self.network = MathNet()
        self.network.load_state_dict(torch.load('./math_net_with_weights_6.pth'))
        self.network.eval()
        self.device = torch.device('cpu')
        self.image1 = Image.new('RGB', (200, 200), 'white')
        self.draw = ImageDraw.Draw(self.image1)
    
class View(tk.Frame):
    """
    View for application. User interface that connects with the controller.
    """
    # Instance Variables
    None
    # Methods
    def __init__(self, parent: tk.Frame) -> None:
        