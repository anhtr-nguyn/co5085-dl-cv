# A1 – CNN vs Transformer trên Food-101

Mốc hiện tại: **A1.1 – Dataset + protocol** (chưa có số liệu cuối). Kết quả so sánh sẽ có ở A1.2 (09/11/2026).

## Dataset: Food-101
- Bossard, Guillaumin, Van Gool. *Food-101 – Mining Discriminative Components with Random Forests*. ECCV 2014. Trang dataset: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- 101 lớp món ăn, 101,000 ảnh JPEG màu: mỗi lớp 750 ảnh train + 250 ảnh test (split chính thức trong `meta/train.txt`, `meta/test.txt`).
- Ảnh đã được tác giả resize về cạnh dài 512 px (100% ảnh có cạnh dài = 512; cạnh ngắn nhỏ nhất 122 px, chỉ 0.014% ảnh có cạnh ngắn < 224).
- Nhãn train **cố ý chưa làm sạch** (có nhiễu: sai nhãn, màu quá đậm); tập test được làm sạch thủ công.
- License (`license_agreement.txt`): ảnh lấy từ Foodspotting, không thuộc sở hữu của ETH Zürich; chỉ được dùng trong phạm vi *scientific fair use*. Đồ án môn học thuộc phạm vi này; repo không phân phối lại ảnh (`data_cache/` nằm trong `.gitignore`).
- Lý do chọn: đủ lớn và khó hơn hẳn E1 (101 lớp, phân biệt chi tiết giữa các món gần giống nhau, cùng một món có hình thức rất khác nhau); ảnh độ phân giải cao phù hợp input 224×224 của backbone pretrained ImageNet; nhãn train nhiễu là điều kiện thực tế để so sánh mức chống chịu của các chế độ train; có sẵn trong `torchvision.datasets.Food101`.

## Chia dữ liệu
| Tập | Nguồn | Số ảnh | Mỗi lớp |
|---|---|---|---|
| Train | split train chính thức, bỏ phần val | 68,175 | 675 |
| Val | 10% split train chính thức, phân tầng theo lớp, `seed=912` | 7,575 | 75 |
| Test | split test chính thức (đã làm sạch) | 25,250 | 250 |

Chỉ số train/val lưu ở `outputs/split.npz` (`python data.py` kiểm tra tái lập). Val dùng để chọn checkpoint và siêu tham số; test chỉ đánh giá một lần ở cuối với best-val checkpoint.

## Tiền xử lý và augmentation
- Chuẩn hóa theo mean/std ImageNet (dùng chung cho mọi chế độ, kể cả from-scratch).
- Train: `RandomResizedCrop(224, scale=(0.08, 1))` → `RandomHorizontalFlip` → `TrivialAugmentWide` → `ToTensor` → `Normalize`.
- Val/test: `Resize(256)` → `CenterCrop(224)`.
- Cùng một pipeline cho mọi run để so sánh công bằng. Label smoothing 0.1 cho mọi run.

## Mô hình (timm)
| Họ | Model | Tham số (head 101 lớp) | Pretrained |
|---|---|---|---|
| CNN | ResNet-50 | 23,714,981 | ImageNet-1k (`resnet50.a1_in1k`) |
| Transformer | DeiT-S/16 | 21,704,549 | ImageNet-1k (`deit_small_patch16_224.fb_in1k`) |

Hai model cùng cỡ tham số và cùng dữ liệu pretrain (ImageNet-1k), nên khác biệt chủ yếu đến từ kiến trúc.

## Ma trận thí nghiệm (8 run)
| Chế độ | Khởi tạo | ResNet-50: phần train | Trainable | DeiT-S: phần train | Trainable |
|---|---|---|---|---|---|
| From-scratch | ngẫu nhiên | toàn bộ | 23.71M (100%) | toàn bộ | 21.70M (100%) |
| Freeze toàn bộ backbone | pretrained | chỉ head `fc` | 0.21M (0.9%) | chỉ head | 0.04M (0.2%) |
| Freeze một phần | pretrained | `layer4` + `fc` (freeze stem, layer1–3) | 15.17M (64%) | block 7–12 + norm + head (freeze patch embed, pos embed, CLS, block 1–6) | 10.69M (49%) |
| Full fine-tune | pretrained | toàn bộ | 23.71M (100%) | toàn bộ | 21.70M (100%) |

Phần bị freeze đặt `requires_grad=False` và giữ ở `eval()` (BatchNorm của ResNet dùng running statistics của ImageNet, không cập nhật).

## Recipe dự kiến
- AdamW, warmup 5% số bước + cosine, weight decay 0.05, batch 128, mixed precision bf16, grad clip 1.0. Training loop tự viết như E1–E3.
- Epoch: freeze toàn bộ 15, freeze một phần và full fine-tune 30, from-scratch 100.
- Learning rate chọn trên val, dự kiến: head-only 1e-3; fine-tune 1e-4 (ResNet) / 5e-5 (DeiT); from-scratch 1e-3 (ResNet) / 5e-4 (DeiT).
- Seed 912 cho cả 8 run; nếu còn thời gian thêm seed 42, 100 cho 6 run pretrained (rẻ hơn nhiều so với from-scratch).
- Ước lượng: khoảng 1 phút/epoch trên một A100 → from-scratch khoảng 1.5–2 giờ/run, toàn bộ ma trận khoảng 6–7 giờ GPU trên 2 × A100.

## Metric và phân tích
- Accuracy top-1 / top-5 và macro-F1 trên val và test.
- Chi phí: số tham số trainable, thời gian/epoch, throughput (ảnh/giây), peak GPU memory.
- Lỗi: confusion matrix 101 lớp, top cặp lớp nhầm nhiều nhất, ảnh sai với độ tự tin cao, lớp có F1 thấp nhất.
- Câu hỏi sẽ trả lời ở A1.2: (1) pretrained giúp ViT nhiều hơn CNN bao nhiêu (ViT thiếu inductive bias cục bộ); (2) khi nào freeze có lợi (linear probe rẻ nhưng bị giới hạn bởi đặc trưng ImageNet; freeze một phần có giữ được accuracy với chi phí thấp hơn không); (3) đánh đổi accuracy – chi phí giữa hai họ.

## Cách chạy
    python data.py     # tải Food-101 vào data_cache/, tạo và kiểm tra split
    python eda.py      # hình EDA -> outputs/figs/, thống kê -> outputs/eda.json

Hình EDA: `outputs/figs/samples.png`, `class_dist.png`, `image_sizes.png`, `augment.png`.
