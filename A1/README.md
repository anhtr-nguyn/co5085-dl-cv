# A1 – CNN vs Transformer trên Food-101

A1.1 (dataset + protocol) và A1.2 (kết quả): 2 kiến trúc × 4 chế độ × 3 seed = 24 run chính, cộng 26 run dò learning rate trên val. Tổng hợp kết quả ở `outputs/results.md`, phân tích lỗi ở `outputs/errors.md`, hình ở `outputs/figs/`.

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

## Recipe (dùng chung cho mọi run, `config/a1.yaml`)
- Training loop tự viết (`train.py`), model từ timm (`models.py`).
- AdamW, weight decay 0.05 (không decay bias, norm, pos embed, CLS token), warmup 5% số bước rồi cosine theo từng bước, grad clip 1.0, label smoothing 0.1, batch 128, bf16 autocast, drop path 0.1.
- Epoch: freeze backbone 15, freeze một phần và full fine-tune 30, from-scratch 100.
- Mỗi epoch đánh giá val, giữ checkpoint có val acc cao nhất; sau khi train xong nạp lại checkpoint đó và đánh giá test **đúng một lần**.
- Seed 912, 42, 100 cho cả 8 cấu hình (chỉ đổi khởi tạo head/model và thứ tự batch; split train/val cố định theo seed 912).

## Chọn learning rate (`outputs/sweep.md`)
Dò trên val, không chạm test (`train.eval_test=false`), seed 912. Head-only dùng lịch đầy đủ 15 epoch; freeze một phần và full fine-tune dùng lịch rút gọn 10 epoch để tiết kiệm thời gian. Lưới ban đầu có 3 giá trị; khi giá trị tốt nhất nằm ở biên trên thì thêm một nấc.

| Model | Chế độ | LR đã thử | LR chọn | Val acc (%) tại LR chọn |
|---|---|---|---|---|
| ResNet-50 | Freeze backbone | 3e-4, 1e-3, 3e-3, 1e-2 | 1e-2 | 66.76 |
| ResNet-50 | Freeze một phần | 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 | 1e-2 | 82.30 |
| ResNet-50 | Full fine-tune | 1e-4, 3e-4, 1e-3, 3e-3 | 1e-3 | 82.85 |
| DeiT-S | Freeze backbone | 3e-4, 1e-3, 3e-3, 1e-2 | 1e-2 | 68.38 |
| DeiT-S | Freeze một phần | 3e-5, 1e-4, 3e-4, 1e-3, 3e-3 | 1e-3 | 84.65 |
| DeiT-S | Full fine-tune | 3e-5, 1e-4, 3e-4, 1e-3 | 3e-4 | 83.97 |

- Ba cấu hình vẫn chọn LR ở biên trên (hai head-only, ResNet freeze một phần), nhưng nấc cuối chỉ tăng 0.1–0.3 điểm so với nấc trước, tức đã gần bão hòa.
- From-scratch **không dò** (100 epoch quá đắt): ResNet-50 1e-3, DeiT-S 5e-4, cố định từ trước. Đây là hạn chế, nhất là với DeiT-S.
- LR cuối nằm ở `config/lr.txt`.

## Kết quả (test, mean ± std qua 3 seed)
| Model | Chế độ | Trainable | Test acc (%) | Top-5 (%) | Macro-F1 (%) | Val acc (%) |
|---|---|---|---|---|---|---|
| ResNet-50 | From-scratch | 23.71M | 84.50 ± 0.16 | 96.14 ± 0.07 | 84.46 ± 0.16 | 79.44 ± 0.15 |
| ResNet-50 | Freeze backbone | 0.21M | 72.14 ± 0.12 | 91.91 ± 0.02 | 71.88 ± 0.12 | 66.81 ± 0.13 |
| ResNet-50 | Freeze một phần | 15.17M | 87.64 ± 0.11 | 97.62 ± 0.08 | 87.62 ± 0.10 | 83.44 ± 0.24 |
| ResNet-50 | Full fine-tune | 23.71M | 88.86 ± 0.05 | 97.94 ± 0.06 | 88.84 ± 0.04 | 84.79 ± 0.20 |
| DeiT-S | From-scratch | 21.70M | 59.53 ± 0.96 | 80.88 ± 0.50 | 59.22 ± 0.93 | 54.73 ± 0.97 |
| DeiT-S | Freeze backbone | 0.04M | 73.28 ± 0.01 | 92.42 ± 0.00 | 73.09 ± 0.01 | 68.38 ± 0.03 |
| DeiT-S | Freeze một phần | 10.69M | 88.80 ± 0.22 | 97.28 ± 0.09 | 88.79 ± 0.22 | 84.76 ± 0.10 |
| DeiT-S | Full fine-tune | 21.70M | 88.89 ± 0.16 | 97.55 ± 0.02 | 88.86 ± 0.16 | 84.73 ± 0.04 |

Bảng đầy đủ (LR, best epoch, thời gian, peak memory) ở `outputs/results.md` / `results.csv`. Test cao hơn val khoảng 4–5 điểm ở mọi cấu hình, vì val lấy từ split train có nhãn nhiễu còn test đã được làm sạch.

### Chi phí (`outputs/bench.md`)
Đo riêng trên một A100 rảnh, batch 128 ảnh ngẫu nhiên, bf16, chỉ tính GPU (không có DataLoader). From-scratch có cùng chi phí mỗi bước với full fine-tune.

| Model | Chế độ | Train (ảnh/s) | Inference (ảnh/s) | Peak mem train (MB) |
|---|---|---|---|---|
| ResNet-50 | Freeze backbone | 6,422 | 6,473 | 891 |
| ResNet-50 | Freeze một phần | 5,298 | 6,463 | 940 |
| ResNet-50 | Full fine-tune | 1,985 | 6,470 | 5,834 |
| DeiT-S | Freeze backbone | 8,284 | 8,357 | 470 |
| DeiT-S | Freeze một phần | 3,854 | 8,382 | 2,492 |
| DeiT-S | Full fine-tune | 2,463 | 8,420 | 4,645 |

Thời gian train thực tế trong `log.json` bị nhiễu: hai job chạy song song và dùng chung 32 CPU cho DataLoader (khoảng 2,400–2,900 ảnh/s). Head-only và freeze một phần bị giới hạn bởi DataLoader (khoảng 28 s/epoch), và ResNet-50 full fine-tune seed 912 mất 38 phút so với 21 phút ở hai seed còn lại vì chạy cùng lúc với run from-scratch. Vì vậy, so sánh chi phí nên dựa vào bảng bench.

### Lỗi (`outputs/errors.md`, `outputs/figs/confusion_top.png`, `error_samples_*.png`)
- Ở cả 8 cấu hình, cặp nhầm nhiều nhất là filet_mignon ↔ steak (12.6–16.7% gộp hai chiều). Các cặp khác lặp lại: chocolate_cake ↔ chocolate_mousse, prime_rib ↔ steak, pork_chop ↔ steak, beef_tartare ↔ tuna_tartare, apple_pie ↔ bread_pudding. Đây là các món cùng màu và kết cấu, khác nhau chủ yếu ở cách cắt hoặc nguyên liệu.
- steak có F1 thấp nhất ở 7/8 cấu hình (DeiT-S from-scratch: apple_pie 0.27). edamame có F1 cao nhất ở cả 8 cấu hình.
- Trong 24 ảnh val sai với độ tự tin cao nhất, một số có vẻ là nhãn nhiễu của split train hoặc ảnh có nhiều món (ví dụ nhãn prime_rib nhưng ảnh là miếng thịt sống, đoán steak). Đây là nhận xét bằng mắt, không đếm định lượng.

## Cách chạy
    python data.py     # tải Food-101 vào data_cache/, tạo và kiểm tra split
    python eda.py      # hình EDA -> outputs/figs/, thống kê -> outputs/eda.json
    python models.py   # kiểm tra số tham số trainable và BatchNorm ở eval() cho phần freeze

    export PY=$(which python)            # python của env co5085
    GPUS="0 1" JOBS=2 PHASE=sweep bash train.sh                    # dò LR -> outputs/sweep/
    python evaluate.py                                             # in LR chọn theo val
    # ghi LR vào config/lr.txt, rồi:
    GPUS="0 1" JOBS=2 PHASE=main SEEDS="912 42 100" bash train.sh  # 24 run -> outputs/runs/
    python bench.py                                                # throughput, memory -> outputs/bench.md
    python evaluate.py                                             # bảng, phân tích lỗi, hình

Một run lẻ: `python train.py --set model.arch=deit_s model.mode=partial train.lr=1e-3 train.epochs=30`. `train.sh` bỏ qua các run đã hoàn tất, nên có thể chạy lại sau khi bị gián đoạn. Trên 2 × A100, sweep mất khoảng 2 giờ và 24 run chính khoảng 6 giờ.

Trong git chỉ giữ `config.yaml`, `log.json` (lịch sử từng epoch) và `test_metrics.json` (gồm per-class metric và confusion matrix 101×101) của mỗi run. Checkpoint và file dự đoán nằm trong `.gitignore`. `evaluate.py` cần `val_preds.pt` cho hình `error_samples_*`; các bảng và hình còn lại chỉ cần các file JSON.

Hình: `outputs/figs/samples.png`, `class_dist.png`, `image_sizes.png`, `augment.png` (EDA); `curves.png`, `acc_by_mode.png`, `acc_vs_cost.png`, `confusion_top.png`, `error_samples_{resnet50,deit_s}_full.png` (kết quả).
