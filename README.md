# CO5085 – Học sâu và ứng dụng trong thị giác máy tính (HK 261)

Nhóm: NTA · Thành viên: Nguyễn Trọng Anh - 2670251
Giảng viên: Lê Thành Sách, Nguyễn Quốc Minh
Trang web: https://anhtr-nguyn.github.io/co5085-dl-cv/

## Cấu trúc repo

| Thư mục | Nội dung | Trạng thái |
|---|---|---|
| `E1/` | Softmax / MLP / CNN trên CIFAR-10 | Hoàn thành |
| `E2/` | MSA tự viết vs PyTorch, tokenization | Hoàn thành |
| `E3/` | LSTM / GRU | Chưa làm |
| `A1/` | CNN vs Transformer trên tập ảnh lớn | Chưa làm |
| `A2/` | Chủ đề tự chọn + phân tích paper | Chưa làm |
| `reports/` | Báo cáo PDF (`exercises.pdf`, `a1.pdf`, `a2.pdf`) | |

## Môi trường

- Python 3.10.22, PyTorch 2.11.0 (CUDA 12.8), torchvision 0.26.0
- GPU: 2 × NVIDIA A100 80GB PCIe (driver 615.71.09), mỗi run dùng 1 GPU
- Cài đặt (conda env `co5085`; `requirements.txt` đã khai báo index CUDA 12.8 của PyTorch):

    conda create -n co5085 python=3.10 -y && conda activate co5085
    pip install -r requirements.txt

## Cách chạy

Xem README trong từng thư mục (`E1/README.md`, ...).

## Tái lập

- Seed, config của các lần chạy trong từng folder.
- Dataset tự tải về `data_cache/` (trong lần đầu chạy).

## Sử dụng công cụ AI

- AI được dùng trong các tác vụ phụ như visualization, refine document README, sửa lỗi ngữ pháp, structure của report, gợi ý cách giải quyết bugs. 
- Các phần liên quan model development, model training, dataloader được viết bởi tác giả. 