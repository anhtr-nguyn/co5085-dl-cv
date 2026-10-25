# E2 – Transformer: MSA tự viết vs PyTorch, các cách tokenize (CIFAR-10)

## Mục tiêu
Hiện thực Multi-Head Self-Attention (MSA) từ đầu và bằng `nn.MultiheadAttention`, đưa vào một ViT nông để phân loại CIFAR-10 (cùng dữ liệu và split với E1), so sánh: (i) MSA tự viết vs PyTorch với cùng tokenizer; (ii) ba cách lấy token.

## Dữ liệu
Giống E1: CIFAR-10, train 40,000 / val 10,000 (`split_seed=912`) / test 10,000. Chuẩn hóa mean/std theo kênh, augmentation RandomCrop(32, padding=4) + RandomHorizontalFlip cho train. Test chỉ dùng một lần ở cuối, chọn checkpoint theo val accuracy.

## Cài đặt
- **MSA tự viết** (`msa.py: MyMSA`): `Linear(C, 3C)` sinh Q, K, V → tách `num_heads` head → `softmax(QKᵀ/√d_head)` → nhân V → ghép head → `Linear(C, C)`. Không dùng `nn.MultiheadAttention`/`nn.TransformerEncoder*`.
- **MSA PyTorch** (`msa.py: TorchMSA`): `nn.MultiheadAttention(batch_first=True)`.
- Kiểm chứng: copy trọng số giữa hai bản cho ra cùng output (sai khác < 1e-5) và cùng gradient, ở mức một lớp MSA (`python msa.py`) và cả mô hình cho 3 tokenizer (`python models.py`).
- **Encoder** (`models.py: MyTinyViT`): tokenizer → [CLS] + positional embedding (học được) → 4 khối Pre-LN `x + MSA(LN(x))`, `x + FFN(LN(x))` (FFN: Linear → GELU → Linear, mlp_ratio 2) → LN → head trên token CLS.

| Tokenizer | Cách lấy token | #Token (+CLS) | #Params |
|---|---|---|---|
| `patch` | Patch 4×4 không chồng lấn, Conv2d(k=4, s=4) chiếu lên d=128 | 64 + 1 | 546,186 |
| `row` | Mỗi hàng ảnh (3×32 giá trị) là một token, Linear(96, 128) | 32 + 1 | 548,234 |
| `cnn` | CNN-stem: 2 × [Conv3×3 → ReLU → MaxPool s2], mỗi vị trí của feature map 8×8 là một token | 64 + 1 | 691,082 |

Số tham số của bản `my` và `torch` bằng nhau với cùng tokenizer.

## Recipe huấn luyện
AdamW (weight decay 0.05, không decay bias/LN/CLS/pos_embed), lr 1e-3, warmup 5 epoch + cosine theo iteration, grad clip 1.0, 100 epoch, batch 128, dropout 0.1 (attention, FFN, embedding). Seed 912, 42, 100, 200, 300 (giống E1). Config: `config/tinyvit.yaml`.

## Cách chạy

    # kiểm tra tính đúng của MSA và mô hình
    python msa.py
    python models.py

    # một run; ghi đè config bằng các KEY=VALUE sau --set
    python train.py --config config/tinyvit.yaml --set model.params.tokenizer_type=row model.params.msa_type=my train.seed=42

    # toàn bộ lưới 3 tokenizer x 2 MSA x 5 seed (bỏ qua cấu hình đã có run)
    GPUS="0 1" JOBS=6 bash train.sh

    # đánh giá trên test + bảng tổng hợp outputs/results.{md,csv}
    python evaluate.py

    # đo tốc độ my vs torch độc lập với training -> outputs/bench.md
    python bench.py

## Kết quả

### Accuracy (mean ± std trên 5 seed, từ `outputs/results.md`)
| Tokenizer | MSA | #Params | Val acc (%) | Test acc (%) | Test macro-F1 (%) | Best epoch | Train acc cuối (%) |
|---|---|---|---|---|---|---|---|
| patch | my | 546,186 | 79.81 ± 0.52 | 80.37 ± 0.59 | 80.30 ± 0.62 | 74–98 | 94.6 |
| patch | torch | 546,186 | 79.84 ± 0.25 | 79.93 ± 0.28 | 79.88 ± 0.25 | 66–95 | 94.4 |
| row | my | 548,234 | 72.09 ± 0.42 | 71.85 ± 0.30 | 71.86 ± 0.31 | 78–96 | 93.2 |
| row | torch | 548,234 | 71.76 ± 0.67 | 72.00 ± 0.51 | 72.04 ± 0.52 | 85–99 | 93.4 |
| cnn | my | 691,082 | 85.52 ± 0.38 | 85.42 ± 0.33 | 85.39 ± 0.32 | 90–100 | 99.3 |
| cnn | torch | 691,082 | 85.48 ± 0.43 | 85.47 ± 0.21 | 85.43 ± 0.21 | 87–96 | 99.3 |

Chênh lệch test acc my − torch theo từng seed (điểm %): patch −0.22…+1.18 (trung bình +0.44), row −0.70…+0.69 (−0.14), cnn −0.46…+0.51 (−0.05).

### Tốc độ (từ `outputs/bench.md`, A100, batch 128, fp32)
| Tokenizer | MSA | #Token | Train (ms/iter) | Inference (img/s) | Peak mem khi train (MB) |
|---|---|---|---|---|---|
| patch | my | 64 | 10.36 | 59,179 | 325 |
| patch | torch | 64 | 10.43 | 69,251 | 267 |
| row | my | 32 | 10.37 | 80,564 | 161 |
| row | torch | 32 | 10.54 | 106,644 | 151 |
| cnn | my | 64 | 11.08 | 47,902 | 692 |
| cnn | torch | 64 | 10.71 | 53,591 | 406 |

Thời gian/epoch trong `outputs/results.md` (3.7–4.1 s) đo khi 6 job chạy song song trên 2 GPU và bị chặn bởi DataLoader, nên không dùng để so sánh tốc độ; bảng trên đo riêng phần model, tuần tự trên một GPU.

### Thử thêm về capacity (patch + torch, 1 seed)
| Cấu hình | #Params | Val acc (%) | Test acc (%) |
|---|---|---|---|
| d128, 4 layer, mlp 2, 100 epoch (baseline, 5 seed) | 546,186 | 79.84 ± 0.25 | 79.93 ± 0.28 |
| d128, 6 layer, mlp 2, 200 epoch | 811,146 | 80.39 | 80.62 |
| d256, 4 layer, mlp 4, 200 epoch | 3,191,562 | 81.80 | 81.91 |

Hình: `outputs/runs/<run>/figs/` (learning curves, confusion matrix val/test, mẫu lỗi), `outputs/figs/compare_curves.png`.

## Nhận xét
- **MSA tự viết vs PyTorch:** accuracy tương đương với cả 3 tokenizer. Chênh lệch trung bình (≤ 0.44 điểm) nhỏ hơn dao động giữa các seed, và dấu chênh lệch đổi theo seed. Điều này khớp với sanity test: hai bản tính cùng một phép toán, khác biệt chỉ đến từ khởi tạo và thứ tự phép tính số thực.
- **Chi phí:** khi train, hai bản nhanh như nhau (khoảng 10.4–11 ms/iter), nhưng bản PyTorch tốn ít bộ nhớ hơn: 406 so với 692 MB với CNN-stem, 267 so với 325 MB với patch. Với `need_weights=False`, `nn.MultiheadAttention` gọi kernel `scaled_dot_product_attention` hợp nhất, không lưu ma trận attention N×N cho backward như bản tự viết. Khi inference, bản PyTorch nhanh hơn 12–32% nhờ fast path hợp nhất cả khối attention.
- **Tokenizer:** CNN-stem (85.4%) > patch (80.1%) > row (71.9%). CNN-stem đưa inductive bias cục bộ của convolution vào trước attention, mỗi token đã tổng hợp thông tin trong một vùng lân cận. Patch 4×4 chỉ chiếu tuyến tính từng ô, nên attention phải tự học quan hệ không gian từ 40k ảnh. Row gộp cả một hàng 32 pixel vào một token, làm mất cấu trúc theo chiều ngang (chỉ còn 32 token), và kém nhất. Row cũng rẻ nhất (151–161 MB, inference nhanh nhất) vì số token ít nhất.
- **Overfitting:** CNN-stem gần như thuộc tập train (99.3%) nhưng vẫn tổng quát tốt nhất. Patch và row dừng ở khoảng 93–95% train acc. Best epoch nằm ở cuối quá trình train với mọi cấu hình, cho thấy cosine schedule + weight decay 0.05 + dropout giữ được val acc tới cuối.
- **Capacity:** tăng model patch lên 6 lần tham số (d256, mlp 4) và gấp đôi số epoch chỉ thêm khoảng 2 điểm (81.9%), vẫn thua CNN-stem nhỏ (0.69M tham số). Nút thắt của ViT ở đây là thiếu inductive bias khi train from scratch trên tập nhỏ, không phải số tham số.
- **So với E1:** ViT tốt nhất (CNN-stem, 85.5%) vẫn thua CNN thuần của E1 (87.75% với cùng augmentation), nhưng chỉ dùng chưa bằng một nửa số tham số (0.69M so với 1.56M). Cặp nhầm nhiều nhất vẫn là cat ↔ dog ở mọi cấu hình (cat có F1 thấp nhất: 0.54 với row, 0.64 với patch, 0.73 với CNN-stem). Row còn nhầm automobile ↔ truck, patch nhầm deer → horse.

## Cấu trúc thư mục
    E2/
    ├── config/tinyvit.yaml
    ├── msa.py  tokenization.py  models.py      MSA, tokenizer, MyTinyViT (+ sanity test)
    ├── data.py  train.py  evaluate.py  visualize.py  bench.py  utils.py  train.sh
    └── outputs/                (chỉ commit log, metric, hình; không commit *.pt)

## Sử dụng AI
Phần sanity test cuối `models.py` (so khớp my vs torch, hoán vị token, gradient) được sinh bằng Claude.ai rồi chỉnh lại.
