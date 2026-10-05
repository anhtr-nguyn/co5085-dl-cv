import torch
import torch.nn as nn

class SoftmaxClassifier(nn.Module):
    def __init__(self, in_dim= 3 * 32 * 32, num_classes = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_dim, num_classes)
    
    def forward(self, x):
        return self.fc(self.flatten(x))

class MLP(nn.Module):
    def __init__(self, in_dim = 3 * 32 * 32, hidden_dim = (512, 256), num_classes = 10, dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]
        prev = in_dim 
        for h in hidden_dim:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(hidden_dim[-1], num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, channels = [32, 64, 128], num_classes = 10):
        super().__init__()
        blocks = []
        prev = 3 #previous dim
        for c in channels:
            blocks += [
                nn.Conv2d(in_channels=prev, out_channels=c, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=c),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ]
            prev = c

        self.feature = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x) # [B, c[-1], 1, 1]
        return self.classifier(x.flatten(1)) # [B, 10]

def build_model(name: str, **kwargs) -> nn.Module:
    registry = {"softmax": SoftmaxClassifier, "mlp": MLP, "cnn": SimpleCNN}
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(registry)}")
    return registry[name](**kwargs)

def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(32, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    criterion = nn.CrossEntropyLoss()


    for name in ["softmax", "mlp", "cnn"]:
        model = build_model(name).to(device)

        # 1. Shape
        out = model(x)
        assert out.shape == (32, 10), out.shape

        # 2. Số tham số
        n = count_params(model)

        # 3. Loss khởi tạo, kỳ vọng quanh ln(10) ~ 2.30
        init_loss = criterion(out, y).item()

        # 4. Overfit 1 batch (nhãn ngẫu nhiên, 32 mẫu) -> loss phải về gần 0
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(200):
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

        # 5. train/eval mode
        model.eval()
        print(f"{name:8s} params={n:>9,}  init_loss={init_loss:.3f}  "
              f"final_loss={loss.item():.4f}  training={model.training}")

    print("OK")