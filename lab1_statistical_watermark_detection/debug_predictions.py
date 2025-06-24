import torch
import numpy as np

# Same SimpleCNN class definition as before
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load watermark data
watermark_data = torch.load('models/watermark_data.pth', weights_only=False)
watermark_inputs = watermark_data['watermark_inputs_noise'][:10]  # Just test 10 samples

# Load and test each model
models = ['baseline', 'very_weak', 'weak', 'medium', 'strong']

for model_name in models:
    if model_name == 'baseline':
        model_path = 'models/baseline_model.pth'
    else:
        model_path = f'models/watermarked_model_{model_name}.pth'
    
    model = SimpleCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()
    
    with torch.no_grad():
        outputs = model(watermark_inputs.to(device))
        _, predictions = torch.max(outputs, 1)
        
    print(f"{model_name.upper()} predictions: {predictions.cpu().numpy()}")
    print(f"Unique predictions: {np.unique(predictions.cpu().numpy())}")
    print()
