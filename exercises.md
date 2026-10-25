---
layout: default
title: E1–E3
---

[← Trang chủ](index.html)

## E1–E3: Bài tập nền tảng

- **Dataset dùng chung:** CIFAR-10 (10 lớp, ảnh màu 32×32). Split cố định (`split_seed=912`): train 40,000 / val 10,000 / test 10,000 (test set gốc, chỉ dùng một lần ở cuối).
- **Code:** [E1](https://github.com/anhtr-nguyn/co5085-dl-cv/tree/main/E1) · [E2](https://github.com/anhtr-nguyn/co5085-dl-cv/tree/main/E2) · [E3](https://github.com/anhtr-nguyn/co5085-dl-cv/tree/main/E3)
- **Báo cáo PDF chung:** [reports/exercises.pdf](reports/exercises.pdf) *(cập nhật: W9 thêm E1+E2, W10 thêm E3)*

| Bài | Nội dung | Trạng thái | Hạn |
|---|---|---|---|
| E1 | Softmax / MLP / CNN | Hoàn thành | 02/11/2026 |
| E2 | MSA tự viết vs PyTorch, tokenization | Hoàn thành | 02/11/2026 |
| E3 | LSTM / GRU | Chưa làm | 09/11/2026 |

## E1

Softmax, MLP và CNN trên ảnh CIFAR-10, training loop tự viết. Adam lr 1e-3, 100 epoch, batch 128, 5 seed (912, 42, 100, 200, 300), mỗi model chạy có và không augmentation (RandomCrop + Flip). Chi tiết: [E1/README.md](https://github.com/anhtr-nguyn/co5085-dl-cv/blob/main/E1/README.md).

| Model | Params | Test acc, không aug (%) | Test acc, có aug (%) |
|---|---|---|---|
| Softmax | 30,730 | 37.36 ± 0.07 | 35.28 ± 0.11 |
| MLP | 9,050,378 | 53.60 ± 0.41 | 49.84 ± 0.30 |
| CNN | 1,558,026 | 83.58 ± 0.25 | 87.75 ± 0.16 |

- CNN vượt xa hai model trên vector flatten dù ít tham số hơn MLP gần 6 lần.
- Augmentation chỉ có lợi cho CNN (+4.2 điểm); với MLP và softmax lại làm giảm accuracy vì model trên vector flatten không bất biến với dịch chuyển.
- Cặp lớp nhầm nhiều nhất ở mọi model là cat ↔ dog.

## E2

MSA tự viết (Linear, nhân ma trận, softmax, ghép head) đối chiếu với `nn.MultiheadAttention`, trong một ViT nông (d=128, 4 khối Pre-LN, token CLS, positional embedding) để phân loại CIFAR-10; so sánh ba cách lấy token: patch 4×4, hàng (row), CNN-stem. AdamW + warmup/cosine, 100 epoch, có augmentation, 5 seed. Chi tiết: [E2/README.md](https://github.com/anhtr-nguyn/co5085-dl-cv/blob/main/E2/README.md).

| Tokenizer | #Token | Params | Test acc, MSA tự viết (%) | Test acc, MSA PyTorch (%) | Peak mem train, tự viết / PyTorch (MB) |
|---|---|---|---|---|---|
| Patch 4×4 | 64 | 546,186 | 80.37 ± 0.59 | 79.93 ± 0.28 | 325 / 267 |
| Row | 32 | 548,234 | 71.85 ± 0.30 | 72.00 ± 0.51 | 161 / 151 |
| CNN-stem | 64 | 691,082 | 85.42 ± 0.33 | 85.47 ± 0.21 | 692 / 406 |

- MSA tự viết và PyTorch cho accuracy tương đương (chênh lệch nhỏ hơn dao động giữa các seed) và tốc độ train như nhau; bản PyTorch ít bộ nhớ hơn và inference nhanh hơn 12–32% nhờ kernel attention hợp nhất.
- CNN-stem > patch > row: inductive bias cục bộ của convolution giúp ViT nhỏ học tốt hơn trên 40k ảnh; row mất cấu trúc theo chiều ngang.
- ViT tốt nhất (85.5%) vẫn thua CNN của E1 (87.75%); tăng model patch lên 3.2M tham số chỉ đạt 81.9%.

## E3

LSTM / GRU trên ảnh dạng chuỗi, so sánh với baseline MLP/CNN của E1. *(Chưa thực hiện.)*
