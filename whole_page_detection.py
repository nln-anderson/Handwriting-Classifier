import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the neural network architecture
class MathNet(nn.Module):
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

# Load the trained model
PATH = './math_net_with_weights_6.pth'
model = MathNet()
model.load_state_dict(torch.load(PATH))
model.eval()  # Set the model to evaluation mode

# Load the image in grayscale
image = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
print("Original Image Loaded:")
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()

# Apply adaptive thresholding
binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
print("Binary Image after Adaptive Thresholding:")
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image after Adaptive Thresholding')
plt.show()

# Advanced noise reduction
binary_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
print("Binary Image after Gaussian Blur:")
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image after Gaussian Blur')
plt.show()

binary_image = cv2.medianBlur(binary_image, 3)
print("Binary Image after Median Blur:")
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image after Median Blur')
plt.show()

# Find contours of the symbols
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total Contours Found: {len(contours)}")

min_contour_area = 100  # Adjust this value based on your images
max_contour_area = 1000  # Upper limit to ignore noise
contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]
print(f"Contours after Filtering by Area: {len(contours)}")

# Filter contours by aspect ratio
aspect_ratio_threshold = 1.5  # Adjust this value based on your symbols
filtered_contours = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    if 1/aspect_ratio_threshold < aspect_ratio < aspect_ratio_threshold:
        filtered_contours.append(cnt)
contours = filtered_contours
print(f"Contours after Filtering by Aspect Ratio: {len(contours)}")

# Extract and resize symbols
symbols = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    symbol_image = binary_image[y:y+h, x:x+w]
    resized_symbol = cv2.resize(symbol_image, (45, 45))
    symbols.append((x, y, w, h, resized_symbol))
    print(f"Symbol extracted at (x: {x}, y: {y}, w: {w}, h: {h})")

# Define the classification function
def classify_symbol(image):
    input_image = image.astype('float32') / 255.0
    input_image = np.expand_dims(input_image, axis=(0, 1))  # Add batch and channel dimensions
    input_tensor = torch.tensor(input_image, dtype=torch.float32)
    
    # Visualize the image fed into the network
    plt.imshow(image, cmap='gray')
    plt.title('Symbol fed into the network')
    plt.show()
    
    with torch.no_grad():
        output = model(input_tensor)
        confidence, predicted_class_idx = torch.max(output, dim=1)
        confidence = confidence.item()
        predicted_class_idx = predicted_class_idx.item()
    print(f"Classified Symbol - Class: {predicted_class_idx}, Confidence: {confidence}")
    return predicted_class_idx, confidence

# Filter symbols with low confidence
confidence_threshold = 0.7  # Adjust this threshold based on your model's performance

classified_symbols = []
for x, y, w, h, symbol in symbols:
    symbol_class, confidence = classify_symbol(symbol)
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
