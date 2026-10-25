import torch 
import torch.nn as nn 

class MyMSA(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        nn.init.xavier_uniform_(self.qkv.weight)
        if qkv_bias:
            nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, num_heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) # [B, num_heads, N, N]
        attn = self.attn_drop(attn) # [B, num_heads, N, N]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x) # [B, N, C]
        return x

class TorchMSA(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True, bias=qkv_bias)

    def forward(self, x):
        return self.mha(x, x, x, need_weights=False)[0]


if __name__ == "__main__":
    torch.manual_seed(0)
    E, H = 128, 4
    my, tm = MyMSA(E, H, qkv_bias=True), TorchMSA(E, H, qkv_bias=True)
    my.eval(); tm.eval()

    with torch.no_grad():
        my.qkv.weight.copy_(tm.mha.in_proj_weight)     # [3E, E]
        my.qkv.bias.copy_(tm.mha.in_proj_bias)
        my.proj.weight.copy_(tm.mha.out_proj.weight)
        my.proj.bias.copy_(tm.mha.out_proj.bias)

        x = torch.randn(4, 17, E)
        print((my(x) - tm(x)).abs().max())             # kỳ vọng ~1e-6
        assert torch.allclose(my(x), tm(x), atol=1e-5)

    print(sum(p.numel() for p in my.parameters()),
        sum(p.numel() for p in tm.parameters()))     