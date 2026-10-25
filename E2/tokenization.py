import torch
import torch.nn as nn

class PatchTokenizer(nn.Module):
    def __init__(self, patch_size, in_channels, embedding_dim, height=32, width=32):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_tokens = (height // patch_size) * (width // patch_size)
        self.projection = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size) # kernel size = patch size, stride = patch size

    def forward(self, x):
        x = self.projection(x) # [B, embedding_dim, H/patch_size, W/patch_size]
        x = x.flatten(2) # [B, embedding_dim, (H/patch_size) * (W/patch_size)]
        x = x.transpose(1, 2) # [B, (H/patch_size) * (W/patch_size), embedding_dim]
        return x    

class RowTokenizer(nn.Module):
    def __init__(self, height, width, in_channels, embedding_dim):
        super().__init__()
        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_tokens = height
        self.projection = nn.Linear(in_channels * width, embedding_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H, W)
        x = x.permute(0, 2, 1, 3) # [B, H, C, W]
        x = x.flatten(2) # [B, H, C * W]
        x = self.projection(x) # [B, H, embedding_dim]
        return x

class CNNStemTokenizer(nn.Module):
    def __init__(self, in_channels, embedding_dim, height=32, width=32):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        layers = [
            nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
        self.layers = nn.Sequential(*layers)
        self.num_tokens = (height // 4) * (width // 4)

    def forward(self, x):
        x = self.layers(x) # [B, embedding_dim, H/2, W/2]
        x = x.flatten(2) # [B, embedding_dim, (H/2) * (W/2)]
        x = x.transpose(1, 2) # [B, (H/2) * (W/2), embedding_dim]   
        return x

def build_tokenizer(name, embed_dim, in_channels=3, img_size=32, patch_size=4):
    if name == "patch":
        return PatchTokenizer(patch_size, in_channels, embed_dim, img_size, img_size)
    if name == "row":
        return RowTokenizer(img_size, img_size, in_channels, embed_dim)
    if name == "cnn":
        return CNNStemTokenizer(in_channels, embed_dim, img_size, img_size)
    raise ValueError(f"Unknown tokenizer '{name}'. Choose from patch/row/cnn")


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 4, 3, 32, 32
    P, E = 4, 128
    pt = PatchTokenizer(P, C, E)
    pt.eval()

    with torch.no_grad():
        x = torch.randn(B, C, H, W)
        print(pt(x).shape, pt.num_tokens) # [B, (H/patch_size) * (W/patch_size), embedding_dim]    

    rt = RowTokenizer(H, W, C, E)
    rt.eval()

    with torch.no_grad():
        x = torch.randn(B, C, H, W)
        print(rt(x).shape, rt.num_tokens) # [B, H, embedding_dim]

    cnn = CNNStemTokenizer(C, E)
    cnn.eval()

    with torch.no_grad():
        x = torch.randn(B, C, H, W)
        print(cnn(x).shape, cnn.num_tokens) # [B, embedding_dim, H/2, W/2]