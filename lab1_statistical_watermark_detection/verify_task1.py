import torch
import torch.nn as nn

# Define the SimpleCNN class directly in this script
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Test if we can load all saved files
try:
    # Load watermark data
    watermark_data = torch.load('models/watermark_data.pth', weights_only=False)
    print("✅ Watermark data loaded successfully")
    print(f"   - Noise samples: {watermark_data['watermark_inputs_noise'].shape}")
    print(f"   - Pattern samples: {watermark_data['watermark_inputs_pattern'].shape}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load baseline model
    baseline_model = SimpleCNN(num_classes=5).to(device)
    baseline_model.load_state_dict(torch.load('models/baseline_model.pth', weights_only=False))
    print("✅ Baseline model loaded successfully")
    
    # Load watermarked models
    model_names = ['very_weak', 'weak', 'medium', 'strong']
    for name in model_names:
        model = SimpleCNN(num_classes=5).to(device)
        model.load_state_dict(torch.load(f'models/watermarked_model_{name}.pth', weights_only=False))
        print(f"✅ {name} watermarked model loaded successfully")
    
    print("\n🎉 All Task 1.1 files are working correctly!")
    print("Ready to proceed to Task 1.2!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Something went wrong with Task 1.1 files")