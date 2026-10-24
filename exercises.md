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
| E2 | MSA tự viết vs PyTorch, tokenization | Đang làm | 02/11/2026 |
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

MSA tự viết (Linear, einsum, softmax, ghép head) đối chiếu với `nn.MultiheadAttention`, trong một encoder nông để phân loại CIFAR-10; so sánh các cách lấy token: patch, hàng (row), CNN-stem. *(Kết quả sẽ cập nhật.)*

## E3

LSTM / GRU trên ảnh dạng chuỗi, so sánh với baseline MLP/CNN của E1. *(Chưa thực hiện.)*
