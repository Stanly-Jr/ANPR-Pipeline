import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- Streamlit Page Config ---
st.set_page_config(page_title="License Plate OCR", layout="centered")


# --- Model Definition ---
class OCR_CNN_Deep(nn.Module):
    def __init__(self, num_classes):
        super(OCR_CNN_Deep, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --- Load Model and Setup (Cached for performance) ---
@st.cache_resource
def load_model():
    model = OCR_CNN_Deep(num_classes=36)
    # Ensure the path below matches exactly where your .pth file is
    model.load_state_dict(torch.load("ocr_model_deep_v1.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


# --- Helper Functions ---
def predict_character(image, model, device):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return class_names[pred.item()]


def extract_characters_with_cca(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    h_img, w_img = image.shape[:2]
    image_area = h_img * w_img

    stats = stats[1:]
    areas = stats[:, cv2.CC_STAT_AREA]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    widths = stats[:, cv2.CC_STAT_WIDTH]

    filtered_areas = areas[areas < 0.25 * image_area]
    if len(filtered_areas) < 2:
        filtered_areas = areas

    Q1, Q3 = np.percentile(filtered_areas, [25, 75])
    IQR = Q3 - Q1
    min_area = max(10, Q1 - 2.5 * IQR)
    max_area = Q3 + 2.0 * IQR
    median_height = np.median(heights)

    characters = []
    for i, (x, y, w, h, area) in enumerate(stats):
        aspect_ratio = w / float(h)
        keep = (
                min_area < area < max_area and
                (0.4 * median_height < h < 2.7 * median_height) and
                (0.15 < aspect_ratio < 2.0)
        )
        if keep:
            char_img = binary[y:y + h, x:x + w]
            density = np.sum(char_img == 0) / (w * h)
            if density < 0.15 or w < 4 or h < 8:
                continue
            char_img = cv2.copyMakeBorder(char_img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
            characters.append((x, y, w, h, char_img))

    if not characters:
        return []

    characters.sort(key=lambda c: c[1] + c[3] // 2)
    line_threshold = median_height * 0.75

    lines = []
    current_line = [characters[0]]
    for i in range(1, len(characters)):
        prev_cy = current_line[-1][1] + current_line[-1][3] // 2
        curr_cy = characters[i][1] + characters[i][3] // 2
        if abs(curr_cy - prev_cy) < line_threshold:
            current_line.append(characters[i])
        else:
            lines.append(current_line)
            current_line = [characters[i]]
    lines.append(current_line)

    lines = [sorted(line, key=lambda c: c[0]) for line in lines]
    return [char_img for line in lines for (_, _, _, _, char_img) in line]


def remove_india_code(chars):
    plate = ''.join(chars)
    if plate.startswith("IND"):
        return chars[3:]
    if plate.startswith("ID"):
        return chars[2:]
    if plate.startswith("ND"):
        return chars[2:]
    if plate.startswith("I") and len(chars) > 1:
        return chars[1:]
    return chars


# --- UI Layout ---
st.title("🚗 Number Plate Recognition")
st.write(
    "Upload an image of a number plate (preferably cropped), and the CNN model will attempt to extract and recognize the characters.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image using PIL, convert to OpenCV format
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert('RGB'))
    opencv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.image(image, caption='Uploaded License Plate', use_column_width=True)
    st.write("Processing image...")

    # Run extraction
    char_images = extract_characters_with_cca(opencv_image)

    if not char_images:
        st.error("No characters detected. Try an image with a clearer license plate.")
    else:
        predictions = []
        for char_img in char_images:
            pred = predict_character(char_img, model, device)
            predictions.append(pred)

        filtered_preds = remove_india_code(predictions)
        license_plate = ''.join(filtered_preds)

        # Show the final predicted text prominently
        st.success("Recognition Complete!")
        st.markdown(f"### **Predicted Plate: {license_plate}**")

        
        st.write("Character Breakdown:")
        fig = plt.figure(figsize=(15, 3))
        for idx, (char_img, pred) in enumerate(zip(char_images, predictions)):
            plt.subplot(1, len(char_images), idx + 1)
            plt.imshow(char_img, cmap='gray')
            plt.title(pred, fontsize=14)
            plt.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
