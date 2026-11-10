# CO5085 – Học sâu và ứng dụng trong thị giác máy tính (HK 261)

Nhóm: NTA · Thành viên: Nguyễn Trọng Anh - 2670251
Giảng viên: Lê Thành Sách, Nguyễn Quốc Minh
Trang web: https://anhtr-nguyn.github.io/co5085-dl-cv/

## Cấu trúc repo

| Thư mục | Nội dung | Trạng thái |
|---|---|---|
| `E1/` | Softmax / MLP / CNN trên CIFAR-10 | Hoàn thành |
| `E2/` | MSA tự viết vs PyTorch, tokenization | Hoàn thành |
| `E3/` | LSTM / GRU trên ảnh dạng chuỗi | Hoàn thành |
| `A1/` | ResNet-50 vs DeiT-S trên Food-101: from-scratch, freeze backbone, freeze một phần, full fine-tune | Hoàn thành (A1.1 + A1.2) |
| `A2/` | Monocular 3D detection trên KITTI, phân tích paper MonoCoP (CVPR 2026) | A2.1 hoàn thành · A2.2 đang thực hiện |
| `reports/` | Báo cáo PDF (hiện có `exercises.pdf`; `a1.pdf`, `a2.pdf` sẽ bổ sung) | |

## Môi trường

- Python 3.10.22, PyTorch 2.11.0 (CUDA 12.8), torchvision 0.26.0
- GPU: 2 × NVIDIA A100 80GB PCIe (driver 615.71.09), mỗi run dùng 1 GPU
- A2 dùng code chính thức của MonoCoP trong một env riêng (Python 3.9, PyTorch 2.4.1, CUDA 12.1), xem `A2/README.md`
- Cài đặt cho E1–E3 và A1 (conda env `co5085`; `requirements.txt` đã khai báo index CUDA 12.8 của PyTorch):

    conda create -n co5085 python=3.10 -y && conda activate co5085
    pip install -r requirements.txt

## Cách chạy

Xem README trong từng thư mục (`E1/README.md`, `E2/README.md`, `E3/README.md`, `A1/README.md`, `A2/README.md`).

## Tái lập

- Mỗi run lưu `config.yaml` (gồm seed) và `log.json` (lịch sử từng epoch) trong `outputs/runs/` của từng thư mục; checkpoint không đưa lên git.
- CIFAR-10 (E1–E3) và Food-101 (A1) tự tải về `data_cache/` ở lần chạy đầu. Trọng số pretrained ImageNet của A1 (timm) tự tải từ Hugging Face Hub.
- KITTI (A2) cần tải thủ công từ trang KITTI (cần đăng ký); xem `A2/README.md`.

## Sử dụng công cụ AI

- AI được dùng trong các tác vụ phụ như visualization, refine document README, sửa lỗi ngữ pháp, structure của report, gợi ý cách giải quyết bugs. 
- Các phần liên quan model development, model training, dataloader được viết bởi tác giả. 