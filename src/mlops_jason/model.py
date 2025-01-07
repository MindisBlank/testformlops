import torch
from torch import nn

class MyAwesomeModel(nn.Module):
    """My awesome model with Batch Normalization."""
    
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2, 2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2, 2)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2, 2)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
