from flask import Flask, render_template, request
import torch
from PIL import Image
import torchvision.transforms as transforms
import os

# Import model + tokenizer
from model import model, tokenizer, device

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Prediction function
def predict(image, text):
    model.eval()

    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        output = model(
            image.unsqueeze(0).to(device),
            inputs['input_ids'].to(device),
            inputs['attention_mask'].to(device)
        )

    pred = torch.argmax(output, dim=1).item()

    return "🚨 Accident Detected" if pred == 0 else "✅ Normal Scene"


# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        text = request.form['text']

        if file:
            # Save image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Load and preprocess image
            image = Image.open(filepath).convert('RGB')
            image = transform(image)

            # Predict
            result = predict(image, text)

            # Send image path to HTML
            image_path = filepath

    return render_template('index.html', result=result, image_path=image_path)


# Run app
if __name__ == '__main__':
    app.run(debug=True)