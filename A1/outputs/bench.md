| Model | Chế độ | Trainable | Train (ms/batch) | Train (ảnh/s) | Inference (ảnh/s) | Peak mem train (MB) |
|---|---|---|---|---|---|---|
| resnet50 | head | 206,949 | 19.9 | 6,422 | 6,473 | 891 |
| resnet50 | partial | 15,171,685 | 24.2 | 5,298 | 6,463 | 940 |
| resnet50 | full | 23,714,981 | 64.5 | 1,985 | 6,470 | 5834 |
| deit_s | head | 38,885 | 15.5 | 8,284 | 8,357 | 470 |
| deit_s | partial | 10,686,437 | 33.2 | 3,854 | 8,382 | 2492 |
| deit_s | full | 21,704,549 | 52.0 | 2,463 | 8,420 | 4645 |

GPU: NVIDIA A100 80GB PCIe, batch 128, bf16 autocast, 50 iters sau 10 warmup. From-scratch có cùng chi phí với full fine-tune.
