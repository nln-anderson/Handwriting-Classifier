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
image = cv2.imread('test_image_processed.jpg', cv2.IMREAD_GRAYSCALE)
print("Original Image Loaded:")
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()

# Preprocess the image: Apply GaussianBlur and thresholding
blurred = cv2.GaussianBlur(image, (5, 5), 0)
_, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print("Thresholded Image:")
plt.imshow(thresholded, cmap='gray')
plt.title('Thresholded Image')
plt.show()

# Find contours of the symbols
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total Contours Found: {len(contours)}")

# Filter contours by area and aspect ratio
min_contour_area = 100  # Adjust this value based on your images
max_contour_area = 10000  # Upper limit to ignore noise
aspect_ratio_threshold = 3.0  # Adjust this value based on your symbols

filtered_contours = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    contour_area = cv2.contourArea(cnt)
    aspect_ratio = float(w) / h
    
    if (min_contour_area < contour_area < max_contour_area and
        1/aspect_ratio_threshold < aspect_ratio < aspect_ratio_threshold):
        filtered_contours.append(cnt)

print(f"Contours after Filtering by Area and Aspect Ratio: {len(filtered_contours)}")

# Draw bounding boxes around the detected contours on the original image
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color image to draw colored boxes
for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

print("Image with Contours:")
plt.imshow(contour_image)
plt.title('Image with Contours')
plt.show()

# Extract and resize symbols
symbols = []
for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    symbol_image = image[y:y+h, x:x+w]
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

