# Application for converting handwriting to LaTeX code

import tkinter as tk
from tkinter import filedialog, Canvas
from torch import nn
import torch.nn.functional as F
import torch
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    model: MathNet

    # Methods
    def __init__(self) -> None:
        self.classes = [
            '!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'C', 'Delta', 'G',
            'H', 'M', 'N', 'R', 'S', 'T', 'X', '[', ']', 'alpha', 'b', 'beta', 'cos', 'd', 'div', 'e', 'exists', 'f',
            'forall', 'forward_slash', 'gamma', 'geq', 'gt', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots',
            'leq', 'lim', 'log', 'lt', 'mu', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'prime', 'q', 'rightarrow', 'sigma',
            'sin', 'sqrt', 'sum', 'tan', 'theta', 'u', 'v', 'w', 'y', 'z', '{', '}'
        ]
        self.model = MathNet()
        self.model.load_state_dict(torch.load('./math_net_with_weights_6.pth'))
        self.model.eval()
        self.device = torch.device('cpu')
        self.image1 = Image.new('RGB', (200, 200), 'white')
        self.draw = ImageDraw.Draw(self.image1)
    
    def find_symbols(self, image: Image) -> list:
        # Preprocess the image
        gray = cv2.cvtColor(image, cv2.COLOR_BAYER_BGGR2GRAY)
        blurred = cv2.GaussianBlur(image, (55, 55), 0)
        _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours of the symbols
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Total Contours Found: {len(contours)}")

        # Filter contours by area and aspect ratio
        filtered_contours = []
        min_area = 90
        image_number = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area:
                x,y,w,h = cv2.boundingRect(c)
                aspect_ratio = w / float(h)
                if 0.3 < aspect_ratio < 10:  # Filter out extremely thin or wide contours
                    # filtered_contours.append((x, y, w, h))
                    # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                    # ROI = image[y:y+h, x:x+w]
                    # cv2.imwrite("ROI_{}.png".format(image_number), ROI)
                    image_number += 1
                    filtered_contours.append(c)

        # Draw bounding boxes around the detected contours on the original image
        contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color image to draw colored boxes
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract and resize symbols
        symbols = []
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            symbol_image = image[y:y+h, x:x+w]
            resized_symbol = cv2.resize(symbol_image, (45, 45))
            symbols.append((x, y, w, h, resized_symbol))
            print(f"Symbol extracted at (x: {x}, y: {y}, w: {w}, h: {h})")
        
        return symbols
    
    def classify_symbol(self, image):
        input_image = image.astype('float32') / 255.0
        input_image = np.expand_dims(input_image, axis=(0, 1))  # Add batch and channel dimensions
        input_tensor = torch.tensor(input_image, dtype=torch.float32)
        
        # Visualize the image fed into the network
        plt.imshow(image, cmap='gray')
        plt.title('Symbol fed into the network')
        plt.show()
        
        with torch.no_grad():
            output = self.model(input_tensor)
            confidence, predicted_class_idx = torch.max(output, dim=1)
            confidence = confidence.item()
            predicted_class_idx = predicted_class_idx.item()
        print(f"Classified Symbol - Class: {predicted_class_idx}, Confidence: {confidence}")
        return predicted_class_idx, confidence

    def classify_all_symbols(self, symbols):
        # Filter symbols with low confidence
        confidence_threshold = 0.7  # Adjust this threshold based on your model's performance

        classified_symbols = []
        for x, y, w, h, symbol in symbols:
            symbol_class, confidence = self.classify_symbol(symbol)
            if confidence > confidence_threshold:
                classified_symbols.append((x, y, w, h, symbol_class))

        # Map class indices to symbols
        symbol_list = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                    '=', 'A', 'C', 'Delta', 'G', 'H', 'M', 'N', 'R', 'S', 'T', 'X', '[', ']', 'alpha', 
                    'b', 'beta', 'cos', 'd', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 'gamma', 
                    'geq', 'gt', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots', 'leq', 'lim', 'log', 
                    'lt', 'mu', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'prime', 'q', 'rightarrow', 'sigma', 'sin', 'sqrt', 
                    'sum', 'tan', 'theta', 'u', 'v', 'w', 'y', 'z', '{', '}']
        symbol_map = {i: symbol_list[i] for i in range(80)}

        # Generate the results
        results = []
        for x, y, w, h, symbol_class in classified_symbols:
            symbol = symbol_map[symbol_class]
            results.append((x, y, symbol))

        # Sort results based on their positions
        results.sort(key=lambda k: (k[1], k[0]))

        # Combine symbols into a readable format
        recognized_text = ''.join([symbol for _, _, symbol in results])
        print("Recognized Text: ", recognized_text)

class View(tk.Frame):
    """
    View for application. User interface that connects with the controller.
    """
    # Instance Variables
    None
    # Methods
    def __init__(self, parent: tk.Frame) -> None:
        super().__init__(parent)
        self.create_layout()

    def create_layout(self) -> None:
        """
        Builds the UI layout
        """
        # Creating widgets
        select_image_but = tk.Button(self, text="Select Image")
        original_image = tk.Label(self)
        detected_image = tk.Label(self)
        convert_but = tk.Button(self, text="Convert to Code")
        code_label = tk.Label(self, text="Code: ")
        code_output_label = tk.Label(self, text="")

        # Assigning instance vars
        self.select_image_but = select_image_but
        self.original_image = original_image
        self.detected_image = detected_image
        self.convert_but = convert_but
        self.code_label = code_label
        self.code_output_label = code_output_label

        # Packing
        select_image_but.grid(row=0, column=0)
        original_image.grid(row=1, column=0)
        detected_image.grid(row=1, column=1)
        convert_but.grid(row=2, column=0)
        code_label.grid(row=3, column=0)
        code_output_label.grid(row=3, column=1)

class Controller:
    """
    Connects the front end and back end operations
    """
    # Instance Variables
    view: View
    model: Model
    
    # Methods
    def __init__(self, view: View, model: Model) -> None:
        self.view = view
        self.model = model

def main() -> None:
    window = tk.Tk()
    window.title("Convert Image to LaTeX Code")

    view = View(window)
    view.pack()

    window.mainloop()

if __name__ == "__main__":
    main()



