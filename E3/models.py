import torch
import torch.nn as nn


class ImageToSequence(nn.Module):
    """[B, C, H, W] -> [B, T, F], không có tham số.

    row:   mỗi hàng là một timestep, T = H,  F = C * W
    col:   mỗi cột là một timestep,  T = W,  F = C * H
    patch: patch p×p không chồng lấn, duyệt theo hàng (raster), T = (H/p)(W/p), F = C * p * p
    """
    def __init__(self, seq_type="row", in_channels=3, img_size=32, patch_size=4):
        super().__init__()
        if seq_type not in ("row", "col", "patch"):
            raise ValueError(f"Unknown seq_type '{seq_type}'. Choose from row/col/patch")
        self.seq_type = seq_type
        self.patch_size = patch_size
        if seq_type == "patch":
            assert img_size % patch_size == 0
            self.seq_len = (img_size // patch_size) ** 2
            self.feat_dim = in_channels * patch_size ** 2
        else:
            self.seq_len = img_size
            self.feat_dim = in_channels * img_size

    def forward(self, x):
        B, C, H, W = x.shape
        if self.seq_type == "row":
            return x.permute(0, 2, 1, 3).reshape(B, H, C * W)     # [B, H, C*W]
        if self.seq_type == "col":
            return x.permute(0, 3, 1, 2).reshape(B, W, C * H)     # [B, W, C*H]
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)                          # [B, H/p, W/p, C, p, p]
        return x.reshape(B, (H // p) * (W // p), C * p * p)


class RNNClassifier(nn.Module):
    """ImageToSequence -> nn.LSTM / nn.GRU -> hidden state của bước cuối -> Linear."""
    def __init__(self, cell="lstm", seq_type="row", hidden_dim=256, num_layers=2,
                 bidirectional=False, dropout=0.1, num_classes=10, img_size=32, patch_size=4):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}
        if cell not in rnn_cls:
            raise ValueError(f"cell must be one of {list(rnn_cls)}")
        self.to_seq = ImageToSequence(seq_type, 3, img_size, patch_size)
        self.rnn = rnn_cls[cell](self.to_seq.feat_dim, hidden_dim, num_layers=num_layers,
                                 batch_first=True, bidirectional=bidirectional,
                                 dropout=dropout if num_layers > 1 else 0.0)
        self.bidirectional = bidirectional
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, num_classes)

    def forward_seq(self, seq):
        """seq [B, T, F] -> logits [B, num_classes]; T có thể ngắn hơn seq_len (dùng khi phân tích)."""
        _, h = self.rnn(seq)
        h = h[0] if isinstance(h, tuple) else h                  # LSTM trả (h_n, c_n)
        # h: [num_layers * num_directions, B, hidden], lấy tầng cuối
        last = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.head(self.drop(last))

    def forward(self, x):
        return self.forward_seq(self.to_seq(x))


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


if __name__ == "__main__":
    torch.manual_seed(0)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))

    # ImageToSequence: kiểm tra từng bước thời gian lấy đúng vùng ảnh
    s = ImageToSequence("row")(x)
    assert s.shape == (32, 32, 96) and torch.equal(s[:, 5], x[:, :, 5, :].reshape(32, -1))
    s = ImageToSequence("col")(x)
    assert s.shape == (32, 32, 96) and torch.equal(s[:, 7], x[:, :, :, 7].reshape(32, -1))
    s = ImageToSequence("patch")(x)
    assert s.shape == (32, 64, 48)
    assert torch.equal(s[:, 9], x[:, :, 4:8, 4:8].reshape(32, -1))   # patch (hàng 1, cột 1)

    for cell in ["lstm", "gru"]:
        for seq in ["row", "col", "patch"]:
            m = RNNClassifier(cell=cell, seq_type=seq)
            out = m(x)
            assert out.shape == (32, 10)
            init_loss = crit(out, y).item()
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            m.train()
            for _ in range(200):
                opt.zero_grad()
                loss = crit(m(x), y)
                loss.backward()
                opt.step()
            print(f"{cell:4s} {seq:5s} T={m.to_seq.seq_len:2d} F={m.to_seq.feat_dim:2d} "
                  f"params={count_params(m):>9,} init={init_loss:.3f} final={loss.item():.4f}")
    print("OK")
