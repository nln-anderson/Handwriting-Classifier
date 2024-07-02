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
    Backend operations of the application. Responsible for storing any data or computational functions.
    """
    def __init__(self):
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

    def preprocess(self, img: Image):
        img = img.resize((45, 45))
        img = ImageOps.grayscale(img)
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
        return img

    def predict(self):
        img = self.image1.convert('L')
        img = np.array(img)
        threshold_value = 128
        img = (img > threshold_value) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img = self.preprocess(img)
        img = img.to(self.device)

        with torch.no_grad():
            output = self.network(img)
            predicted_class_idx = torch.argmax(output).item()
            predicted_class_name = self.classes[predicted_class_idx]

        return predicted_class_name

class View(tk.Frame):
    """
    User interface for the application. Contains all the widgets and frames. It is a child class of the tk.Frame class.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.create_layout()

    def create_layout(self):
        canvas_size = 200
        self.canvas = Canvas(self, width=canvas_size, height=canvas_size, bg='white')
        self.canvas.pack()

        self.clear_button = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side="left")

        self.predict_button = tk.Button(self, text="Predict", command=self.predict)
        self.predict_button.pack(side="left")

        self.result_label = tk.Label(self, text="Draw a symbol and click Predict")
        self.result_label.pack(side="left")

    def clear_canvas(self):
        self.canvas.delete("all")

    def predict(self):
        pass  # This will be set by the controller

class Controller:
    """
    Connects the view and controller. Responsible for binding button presses and fetching from the model to update the view.
    """
    def __init__(self, model: Model, view: View):
        self.model = model
        self.view = view
        self.view.canvas.bind("<B1-Motion>", self.paint)
        self.view.clear_button.configure(command=self.clear_canvas)
        self.view.predict_button.configure(command=self.predict)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.view.canvas.create_oval(x1, y1, x2, y2, fill='black', width=1)
        self.model.draw.line([x1, y1, x2, y2], fill='black', width=2)

    def clear_canvas(self):
        self.view.clear_canvas()
        self.model.image1 = Image.new('RGB', (200, 200), 'white')
        self.model.draw = ImageDraw.Draw(self.model.image1)

    def predict(self):
        prediction = self.model.predict()
        self.view.result_label.config(text=f"Predicted: {prediction}")

def main():
    window = tk.Tk()
    window.title("Symbol Recognition")

    model = Model()
    view = View(window)
    controller = Controller(model, view)

    view.pack(side="top")

    window.mainloop()

if __name__ == "__main__":
    main()
