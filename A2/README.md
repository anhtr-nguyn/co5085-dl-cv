# A2 – Monocular 3D Object Detection trên KITTI (MonoCoP)

Mốc: **A2.1 + A2.2 hoàn thành.** Báo cáo: [`reports/a2.pdf`](../reports/a2.pdf) · trang A2: https://anhtr-nguyn.github.io/co5085-dl-cv/a2.html

Tái hiện MonoCoP (CVPR 2026) bằng code chính thức của tác giả và so sánh với một baseline dự đoán song song cài trên **cùng code, cùng recipe**, chỉ khác các head 3D. Thêm ablation chọn nhánh khi suy luận, sai số theo khoảng cách và độ bền trên KITTI-C.

## Kết quả chính

KITTI val, Car, AP3D|R40 IoU 0.7, mức Moderate, checkpoint best trên val (như paper). Bảng đầy đủ: [`outputs/results.md`](outputs/results.md).

| Biến thể | Run | Trung bình ± std | Median | Epoch 250 (trung bình) |
|---|---|---|---|---|
| MonoCoP (CoP + GS) | 5 | 23.75 ± 0.35 | 23.80 | 22.56 |
| Baseline song song | 3 | 23.60 ± 1.00 | 23.89 | 22.43 |
| Paper (MonoCoP) | 5 | – | 23.98 | – |

- Tái hiện được con số của paper trong phạm vi 0.2 điểm, nhưng CoP + GS không cải thiện vượt dao động giữa các run so với baseline cùng code (paper: +2.86 so với baseline 21.12, là MonoDGP do tác giả chạy lại).
- GS với ngưỡng mặc định (`uncertainty_tau = 5` trên log σ) chọn nhánh chain cho 100% object; mọi quy tắc chọn nhánh khác cho 23.70–23.79, và GS chọn đúng nhánh tốt hơn cho 49.4% xe (paper: 82.18%).
- Checkpoint best cao hơn epoch 250 trung bình 1.18 điểm vì được chọn trên chính tập báo cáo.

## Paper phân tích

Zhihao Zhang, Abhinav Kumar, Girish Chandar Ganesan, Xiaoming Liu. **Unleashing the Power of Chain-of-Prediction for Monocular 3D Object Detection**. CVPR 2026 (Highlight), trang 18755–18765. [arXiv:2505.04594](https://arxiv.org/abs/2505.04594) · code chính thức: [alanzhangcs/MonoCoP](https://github.com/alanzhangcs/MonoCoP).

```bibtex
@inproceedings{zhang2026monocop,
  title     = {Unleashing the Power of Chain-of-Prediction for Monocular 3D Object Detection},
  author    = {Zhang, Zhihao and Kumar, Abhinav and Ganesan, Girish Chandar and Liu, Xiaoming},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

Phân tích đầy đủ (phương pháp, thí nghiệm, đối chiếu paper với code, hạn chế) ở mục 3 của báo cáo.

## Nội dung thư mục

| File | Vai trò |
|---|---|
| `eda.py` | Thống kê KITTI train/val và hình EDA → `outputs/eda.json`, `outputs/figs/` |
| `monocop/monocop.patch` | Toàn bộ thay đổi so với MonoCoP commit `7d39324`: bản vá build CUDA 12.1/A100, cờ `chain_prediction` (baseline song song) trong `monocop.py`, `freeze_at=-1` |
| `monocop/monocop_car_parallel.yaml` | Cấu hình baseline song song, giống `monocop_car.yaml` trừ cờ `chain_prediction: False`, tên model và thư mục output |
| `monocop/run_queue.sh` | Chạy tuần tự nhiều run train trên một GPU |
| `monocop/analyze.py` | Chỉ suy luận: ablation chọn nhánh (một lần forward, giải mã mọi quy tắc offline), MAE theo khoảng cách, KITTI-C, latency/FPS |
| `monocop/run_analysis.sh` | Chạy `analyze.py` cho nhiều checkpoint trên một GPU |
| `summarize.py` | Đọc log train và `outputs/analysis/*.json` → `outputs/results.json`, `outputs/results.md`, hình kết quả, bảng LaTeX cho báo cáo |
| `outputs/analysis/*.json` | Kết quả `analyze.py` cho từng checkpoint (9 run) |

## Dữ liệu: KITTI 3D Object Detection

- Geiger, Lenz, Urtasun. *Are we ready for autonomous driving? The KITTI vision benchmark suite*. CVPR 2012. https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
- Dùng ảnh camera trái (`image_2`), ma trận chiếu `P2` trong `calib/`, nhãn 3D trong `label_2/`; không dùng LiDAR hay dữ liệu bổ sung. License CC BY-NC-SA 3.0; repo không phân phối lại ảnh.
- Split train/val 3,712/3,769 ảnh theo Chen et al. (CVPR 2016). Root cần có `ImageSets/{train,val}.txt` và `training/{image_2,label_2,calib}/`.
- KITTI-C: 13 loại nhiễu ở mức 1 trên toàn bộ ảnh val (Lin et al., *MonoTTA: Fully Test-Time Adaptation for Monocular 3D Object Detection*, ECCV), đặt tại `<KITTI>/kitti_c/<corruption>_1/`.

| Tập | Số ảnh | Car | Pedestrian | Cyclist |
|---|---|---|---|---|
| train | 3,712 | 14,357 | 2,207 | 734 |
| val | 3,769 | 14,385 | 2,280 | 893 |

## Cách chạy

**1. Chuẩn bị MonoCoP** (env riêng: Python 3.9, PyTorch 2.4.1, CUDA 12.1; cần `nvcc` để build op deformable attention):

    git clone https://github.com/alanzhangcs/MonoCoP && cd MonoCoP
    git checkout 7d39324
    git apply /path/to/co5085-dl-cv/A2/monocop/monocop.patch
    cp /path/to/co5085-dl-cv/A2/monocop/monocop_car_parallel.yaml config/
    cd lib/models/monocop/ops && bash make.sh && cd -
    # sửa dataset.root_dir trong config/*.yaml trỏ tới KITTI

**2. Train** (từ thư mục gốc MonoCoP; mỗi run khoảng 15 giờ trên một A100):

    export NUMBA_CUDA_USE_NVIDIA_BINDING=1     # thiếu biến này, evaluator numba bị segfault trên máy này
    CUDA_VISIBLE_DEVICES=0 python train_val.py --config config/monocop_car.yaml --seed 42
    CUDA_VISIBLE_DEVICES=1 python train_val.py --config config/monocop_car_parallel.yaml --seed 42
    # hoặc một hàng đợi trên một GPU
    bash /path/to/A2/monocop/run_queue.sh 0 config/monocop_car.yaml:42 config/monocop_car.yaml:100

**3. Phân tích** (khoảng 25 phút mỗi checkpoint với `--sets all`):

    python /path/to/A2/monocop/analyze.py --config config/monocop_car.yaml \
        --ckpt outputs/monocop_car/<run>/checkpoint_best.pth \
        --out /path/to/A2/outputs/analysis/<run>.json --sets all --speed

**4. Tổng hợp** (env `co5085`):

    python A2/summarize.py --monocop-root /path/to/MonoCoP
    python A2/eda.py --root /path/to/KITTI        # thống kê EDA

## Ghi chú tái lập

- Op deformable attention cộng dồn gradient bằng `atomicAdd`, nên hai run cùng seed vẫn khác nhau (ba run MonoCoP seed 444: 23.55 / 23.95 / 23.80).
- Run MonoCoP 01/10 giữ lớp stem của backbone đóng băng (`freeze_at=0` như code gốc); mọi run khác dùng `freeze_at=-1`. Run này được báo cáo riêng.
- Chạy hai run trên cùng một A100 làm tổng tốc độ giảm khoảng 11%, nên các run được chạy tuần tự.
- Log train, checkpoint và output đánh giá nằm ngoài repo; AP theo từng epoch của mọi run được trích vào `outputs/results.json`.
