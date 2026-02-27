from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import io
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify

app = Flask(__name__)

# SQLite local file database config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

# Remove the @app.before_first_request decorator and function

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Missing username or password'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'message': 'Username already exists'}), 409

    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password_hash=hashed_password)

    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 200

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Missing username or password'}), 400

    user = User.query.filter_by(username=username).first()

    if user and check_password_hash(user.password_hash, password):
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'message': 'Invalid username or password'}), 401



# Your DenseNet201Model class exactly as in training
class DenseNet201Model(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet201Model, self).__init__()
        from torchvision import models
        self.densenet = models.densenet201(pretrained=False)
        self.densenet.classifier = nn.Sequential(
            nn.Linear(1920, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.densenet(x)


from torchvision import transforms

# Define transform to preprocess input images (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# Number of classes
num_classes = 8

# Your class names in the same order as training dataset classes
class_names = {
    0: 'dyed-lifted-polyps',
    1: 'dyed-resection-margins',
    2: 'esophagitis',
    3: 'normal-cecum',
    4: 'normal-pylorus',
    5: 'normal-z-line',
    6: 'polyps',
    7: 'ulcerative-colitis',
}

# Instantiate your model with correct number of classes
model = DenseNet201Model(num_classes=num_classes)
model.load_state_dict(torch.load('Stomachcancer.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    input_tensor = transform(image).unsqueeze(0)  # Add batch dim

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()

    predicted_class = class_names.get(class_idx, 'Unknown')

    return jsonify({'disease_name': predicted_class})


if __name__ == '__main__':
    # Create tables explicitly before running app
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000)
