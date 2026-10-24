# E1 – Phân loại ảnh: Softmax, MLP, CNN (CIFAR-10)

## Mục tiêu
So sánh ba bộ phân loại trên CIFAR-10: softmax trên vector flatten, MLP và CNN.
Training loop tự viết (forward → loss → backward → step).

## Dữ liệu
- CIFAR-10: 10 lớp, ảnh màu 32×32, 50,000 train + 10,000 test.
- Split: train 40,000 / val 10,000 (hoán vị ngẫu nhiên cố định bằng `split_seed=912`, `python data.py` lưu chỉ số ra `outputs/split.npz`) / test 10,000.
- Chuẩn hóa mean/std theo kênh (mean = (0.491, 0.482, 0.446), std = (0.247, 0.243, 0.261)).
- Augmentation (khi `data.augment=true`, chỉ áp cho train): RandomCrop(32, padding=4) + RandomHorizontalFlip.
- Test set chỉ dùng một lần ở cuối (`evaluate.py`), chọn epoch bằng val.

## Mô hình
| Model | Kiến trúc | Tham số |
|---|---|---|
| Softmax | Flatten → Linear(3072, 10) | 30,730 |
| MLP | 3072 → 2048 → 1024 → 512 → 256 → 10 (ReLU, không dropout) | 9,050,378 |
| CNN | 4 × [Conv3×3 → BN → ReLU → MaxPool] (64/128/256/512) + GAP + Linear(512, 10) | 1,558,026 |

## Recipe huấn luyện (chung cho cả 3 model)
Adam, lr 1e-3, 100 epoch, batch size 128, weight decay 0, không scheduler, CrossEntropyLoss.
Seed: 912, 42, 100, 200, 300 (`torch.use_deterministic_algorithms(True)`, chạy lại cùng seed cho kết quả giống hệt).
Checkpoint lưu theo val accuracy tốt nhất.

## Cách chạy

    # huấn luyện; mỗi run ghi vào outputs/runs/<model>_aug<0|1>_s<seed>_<datetime>/
    # (config.yaml, log.json, best.pt, val_preds.pt, figs/)
    python train.py --config config/softmax.yaml
    python train.py --config config/mlp.yaml
    python train.py --config config/cnn.yaml

    # ghi đè config: nhiều KEY=VALUE đặt sau MỘT lần --set
    python train.py --config config/cnn.yaml --set data.augment=true train.seed=42

    # toàn bộ lưới 3 model x {không aug, aug} x 5 seed (bỏ qua cấu hình đã có run);
    # log stdout từng run ở outputs/logs/
    bash train.sh                     # tuần tự, GPU 0
    GPUS="0 1" JOBS=6 bash train.sh   # song song trên 2 GPU

    # đánh giá trên test (test_metrics.json, test confusion matrix cho từng run),
    # bảng tổng hợp outputs/results.{md,csv}, đường cong so sánh outputs/figs/compare_curves.png
    python evaluate.py                       # mọi run trong outputs/runs
    python evaluate.py outputs/runs/<run> ...  # chỉ các run chỉ định

    # vẽ lại hình cho một run (curves, confusion matrix val, mẫu lỗi)
    python visualize.py outputs/runs/<run>

    # kiểm tra split + lưu ảnh mẫu outputs/figs/samples.png
    python data.py

## Kết quả
| Model | Aug | Params | Best val acc (%) | Test acc (%) | Test macro-F1 (%) | Best epoch | Train acc cuối (%) | Peak mem (MB) |
|---|---|---|---|---|---|---|---|---|
| Softmax | không | 30,730 | 37.41 ± 0.11 | 37.36 ± 0.07 | 36.67 ± 0.09 | 7 | 44.6 | 19 |
| Softmax | có | 30,730 | 35.84 ± 0.05 | 35.28 ± 0.11 | 34.52 ± 0.11 | 40 | 30.0 | 19 |
| MLP | không | 9,050,378 | 53.74 ± 0.10 | 53.60 ± 0.41 | 53.63 ± 0.47 | 15–92 | 97.5 | 192 |
| MLP | có | 9,050,378 | 49.38 ± 0.31 | 49.84 ± 0.30 | 49.52 ± 0.25 | 60–85 | 69.2 | 192 |
| CNN | không | 1,558,026 | 83.97 ± 0.09 | 83.58 ± 0.25 | 83.54 ± 0.26 | 26–76 | 99.9 | 212 |
| CNN | có | 1,558,026 | 88.34 ± 0.18 | 87.75 ± 0.16 | 87.70 ± 0.18 | 83–99 | 98.5 | 212 |

(mean ± std trên 5 seed, 100 epoch, một GPU NVIDIA A100 80GB. Số liệu lấy từ `outputs/runs/*/log.json` và `test_metrics.json`, tổng hợp trong `outputs/results.md`.)

Thời gian: khi chạy một mình, một epoch mất khoảng 2.1–2.2 s (không aug) và 2.9 s (có aug) cho cả ba model, tức khoảng 210–290 s cho 100 epoch. Thời gian bị chặn bởi DataLoader trên CPU chứ không phải bởi model. Cột thời gian trong `outputs/results.md` của các nhóm MLP không aug, Softmax/CNN có aug (trừ seed 912) cao hơn vì các run này chạy song song 6 job trên 2 GPU, không dùng để so sánh tốc độ.

Hình: `outputs/runs/<run>/figs/` (learning curves, confusion matrix val/test, mẫu lỗi), `outputs/figs/compare_curves.png`, `outputs/figs/samples.png`.

## Nhận xét ngắn
- CNN vượt xa hai model trên vector flatten (83.6% so với 53.6% và 37.4%, cùng điều kiện không aug) dù ít tham số hơn MLP gần 6 lần: conv + pooling khai thác tính cục bộ và chia sẻ trọng số, còn flatten làm mất cấu trúc không gian.
- Augmentation (crop + flip) chỉ có lợi cho CNN: +4.2 điểm (83.6% → 87.8%), giảm overfit (train acc 100% → 98.5%, best epoch dời từ 26–76 về 83–99). Với MLP và softmax, augmentation làm giảm accuracy (−3.8 và −2.1 điểm): model trên vector flatten không bất biến với dịch chuyển, nên ảnh bị crop lệch vài pixel trở thành một vector khác hẳn, khiến bài toán khó hơn chứ không đóng vai trò regularization (train acc MLP tụt từ 97.5% xuống 69%).
- MLP không aug overfit mạnh (train 97.5%, val 53.7%); softmax underfit (train khoảng 45%, best val sớm ở epoch 7 rồi val loss tăng nhẹ).
- Cặp lớp nhầm nhiều nhất ở mọi model là cat ↔ dog; cat có F1 thấp nhất (0.70 với CNN không aug, 0.76 có aug, 0.36 với MLP, 0.20 với softmax). Model flatten còn nhầm theo màu và nền: softmax đoán frog cho nhiều ảnh deer/cat, nhầm airplane → ship (nền trời/biển xanh) khi có aug; MLP nhầm automobile ↔ truck.

## Cấu trúc thư mục
    E1/
    ├── config/{softmax,mlp,cnn}.yaml
    ├── data.py  models.py  train.py  evaluate.py  visualize.py  utils.py  train.sh
    └── outputs/                (không commit, nằm trong .gitignore)
        ├── runs/<run>/         config.yaml, log.json, test_metrics.json, figs/, best.pt
        ├── figs/               compare_curves.png, samples.png
        └── results.{md,csv}
