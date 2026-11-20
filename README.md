# ZS-BERT — tái lập trên Kaggle Free

Tái lập ZS-BERT trong Table 2 (NAACL 2021): FewRel và Wiki-ZSL, m=5/10/15,
5 seed 300–304, 10 epoch/run. 30 runs có thể cần hơn 30 phiên 12 giờ.

## Bắt đầu

1. Đọc [kế hoạch](docs/IMPLEMENTATION_PLAN.md) và [hướng dẫn từng run](docs/KAGGLE_SINGLE_RUN.md).
2. Push bộ code này trước khi dùng notebook mới (notebook tải scripts từ repo GitHub).
3. Import một notebook, GPU + Internet, chọn DATASET và RUN_ID.
4. Chạy smoke test cho cả hai dataset và kiểm tra pause/resume trước runs chính thức.
5. Khóa CODE_REV và checksum dữ liệu đã kiểm tra; SMOKE_TEST=False; Save & Run All.
6. Download evidence ZIP và notebook đã chạy; nhập kết quả bằng scripts/collect_results.py.

- [m=5](notebooks/kaggle/zs-bert-m-5.ipynb)
- [m=10](notebooks/kaggle/zs-bert-m-10.ipynb)
- [m=15](notebooks/kaggle/zs-bert-m-15.ipynb)
- [Kết quả và tiến độ](notebooks/kaggle/RESULTS.md)
- [CSV từng run](notebooks/kaggle/results.csv)

## Cấu trúc

- model/: model/loss, dataset và metric implementation hiện có.
- scripts/kaggle_run.py: tải dữ liệu/model, kiểm tra config, điều phối một seed, đóng gói bằng chứng.
- scripts/train_run.py: train/evaluate, checkpoint và deadline.
- scripts/normalize_data.py và build_attributes.py: tiền xử lý và thuộc tính cố định.
- scripts/collect_results.py: kiểm tra/nhập bằng chứng và tổng hợp đủ 5 seed.
- notebooks/kaggle/: 3 notebook, results.csv, RESULTS.md; runs/ chứa evidence từng attempt.
- notebooks/kaggle/executed/: notebook có output tải từ version Kaggle.
- notebooks/kaggle/historical/: metric trích log của bộ chạy cũ, chưa gộp vào bộ mới.
- notebooks/legacy/: notebook cũ giữ nguyên làm bằng chứng lịch sử.
- docs/: hướng dẫn, kế hoạch, bài báo và giới hạn tái lập.
- resources/: tài nguyên implementation cũ; runner mới dùng snapshot mô tả tác giả có checksum.
- data/raw/: dữ liệu local không commit; notebook mới tự tải link tác giả.

## Trạng thái và giới hạn

Bộ refactor chưa được chạy trên GPU Kaggle. Hai run FewRel seed 300 lịch sử đã
hoàn thành theo log, không được tự trộn vào bộ cấu hình mới. Dữ liệu local FewRel:
56.000/80 quan hệ; Wiki-ZSL: 93.383/113, khác Table 1 (94.383). Runner dừng trước
train nếu lệch count, trừ khi người dùng cho phép và ghi rõ giải trình.

Dùng FP32, batch 4, Adam; theo implementation BERT-large 1024. Giữ quy tắc entity
mask token gốc+1 (chưa căn chỉnh WordPiece). Main metric theo tác giả, có sklearn
macro bổ sung. Best chọn epoch bằng test F1; final là epoch 10. Không hứa trùng điểm báo.

requirements-original.txt giữ bộ thư viện tác giả để tham khảo;
requirements-kaggle.txt dùng runtime tương thích hiện đại. Ghi môi trường thực tế mỗi run.
Xem [chi tiết giới hạn](docs/REPRODUCIBILITY.md).

Nguồn: [bài báo](https://aclanthology.org/2021.naacl-main.272/),
[code tác giả](https://github.com/dinobby/ZS-BERT),
[repo dự án](https://github.com/Student-persist-1919/zs-bert-simulation).
