import timm
import torch.nn as nn

from data import NUM_CLASSES

ARCHS = {
    "resnet50": "resnet50.a1_in1k",
    "deit_s": "deit_small_patch16_224.fb_in1k",
}
MODES = ("scratch", "head", "partial", "full")


def trainable_prefixes(arch, mode):
    """Tiền tố tên tham số được train; None = train toàn bộ."""
    if mode in ("scratch", "full"):
        return None
    if arch == "resnet50":
        return ("fc.",) if mode == "head" else ("layer4.", "fc.")
    if arch == "deit_s":
        if mode == "head":
            return ("head.",)
        # 12 block, freeze patch_embed / pos_embed / cls_token / block 0-5
        return tuple(f"blocks.{i}." for i in range(6, 12)) + ("norm.", "head.")
    raise ValueError(f"Unknown arch '{arch}'")


def build_model(arch, mode, drop_path=0.0):
    """
    arch: resnet50 | deit_s
    mode: scratch (khởi tạo ngẫu nhiên, train toàn bộ)
          head    (pretrained, chỉ train head)
          partial (pretrained, train nửa sau theo độ sâu + head)
          full    (pretrained, train toàn bộ)
    Head luôn là Linear(·, 101) khởi tạo ngẫu nhiên.
    """
    if arch not in ARCHS or mode not in MODES:
        raise ValueError(f"arch {arch} / mode {mode} không hợp lệ")
    model = timm.create_model(ARCHS[arch], pretrained=mode != "scratch",
                              num_classes=NUM_CLASSES, drop_path_rate=drop_path)
    prefixes = trainable_prefixes(arch, mode)
    if prefixes is not None:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(prefixes)
    return model


def frozen_modules(model):
    """Các module không có tham số nào được train; giữ ở eval() khi train (BN dùng thống kê ImageNet)."""
    out = []
    for m in model.modules():
        params = list(m.parameters(recurse=False))
        if params and not any(p.requires_grad for p in params):
            out.append(m)
    return out


def set_train_mode(model):
    model.train()
    for m in frozen_modules(model):
        m.eval()


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


if __name__ == "__main__":
    import torch
    x = torch.randn(2, 3, 224, 224)
    for arch in ARCHS:
        for mode in MODES:
            m = build_model(arch, mode)
            set_train_mode(m)
            n_bn_train = sum(1 for mod in m.modules()
                             if isinstance(mod, nn.BatchNorm2d) and mod.training)
            assert m(x).shape == (2, NUM_CLASSES)
            tr, tot = count_params(m), count_params(m, False)
            print(f"{arch:8s} {mode:8s} trainable {tr:>11,} / {tot:,} ({100*tr/tot:5.1f}%)"
                  f"  BN đang train: {n_bn_train}")
