# ZS-BERT — tái lập trên Kaggle Free

2 datasets (FewRel/Wiki-ZSL) × 3 cấu hình m (5/10/15) × 5 seed (300–304):
30 runs, mỗi run đủ 10 epoch. Có thể cần hơn 30 phiên Save & Run All do giới hạn 12 giờ.

## Chạy chính thức

1. Tải một trong ba notebook bên dưới và import vào Kaggle.
2. GPU + Internet On; chỉ đổi DATASET và RUN_ID trong cell cấu hình.
3. Giữ SMOKE_TEST=False, code revision và checksum đã khóa; Save Version → Save & Run All.
4. Sau mỗi run, tải ZIP bằng chứng và notebook có output. Xem [hướng dẫn](docs/KAGGLE_SINGLE_RUN.md).
5. Nếu paused, giữ full output trên Kaggle để tiếp tục cùng run. Chỉ complete đủ 10 epoch mới được tổng hợp.

- [Notebook m=5](notebooks/kaggle/zs-bert-m-5.ipynb)
- [Notebook m=10](notebooks/kaggle/zs-bert-m-10.ipynb)
- [Notebook m=15](notebooks/kaggle/zs-bert-m-15.ipynb)
- [Kết quả](notebooks/kaggle/RESULTS.md) và [CSV từng run](notebooks/kaggle/results.csv)

RUN_ID=1/2/3/4/5 tương ứng seed=300/301/302/303/304.
Mỗi notebook dùng được cho cả DATASET="fewrel" và DATASET="wikizsl".

## Cấu trúc

```text
BTL/
├── notebooks/
│   ├── kaggle/          # Chỉ 3 notebook chính, results.csv, RESULTS.md
│   └── archive/         # Notebook implementation cũ, giữ truy nguyên
├── experiments/
│   ├── runs/            # Evidence chính thức theo run/attempt (tạo khi nhập ZIP)
│   ├── executed/        # Notebook đã chạy, có output
│   ├── validation/      # Bằng chứng smoke test đã đạt
│   └── historical/      # Log/kết quả lịch sử và các lần lỗi
├── model/               # Model, loss, dataset, metric
├── scripts/             # Điều phối Kaggle, train, dữ liệu và tổng hợp
├── resources/datasets/  # Snapshot JSON nén và SHA256
├── docs/                # Kế hoạch, hướng dẫn, giới hạn và bài báo
├── tests/               # Kiểm tra local
└── data/raw/            # Dữ liệu local, không commit
```

## Trạng thái và giới hạn

Smoke test train/evaluate/export đã đạt trên T4 cho cả FewRel và Wiki-ZSL;
[bằng chứng](experiments/validation/README.md) và [notebook có output](experiments/executed/).
Theo yêu cầu người dùng, bỏ qua kiểm thử GPU pause/resume riêng để chạy chính thức.
Cơ chế checkpoint/deadline vẫn được giữ. FewRel m=10 seed 300 đã resume đầu epoch 10
và hoàn tất trên T4 theo notebook có output; đang chờ ZIP để nhập results.csv.
Resume giữa epoch (m=5) chưa xác minh. Bảng kết quả hiện chưa nhập full run.

Snapshot FewRel: 56.000 mẫu/80 quan hệ. Snapshot Wiki-ZSL: 93.383/113, khác
Table 1 (94.383). Notebook cho phép chạy snapshot Wiki-ZSL và ghi rõ discrepancy
trong provenance; không khẳng định tái lập đúng dataset của báo. Các file BTL chưa
được xác minh giống hệt file Google Drive tác giả.

Theo implementation hiện có: BERT-large 1024, FP32, batch 4, Adam, lr 5e-6,
gamma 7.5, alpha 0.4. Giữ entity mask token gốc+1 (chưa căn chỉnh WordPiece).
Main metric theo tác giả; thêm sklearn macro. Best chọn bằng test F1, final là
 epoch 10. Xem [giới hạn tái lập](docs/REPRODUCIBILITY.md).

requirements-original.txt giữ dependencies tác giả để tham khảo;
requirements-kaggle.txt dùng môi trường hiện đại, lưu phiên bản thực tế mỗi run.
[Kế hoạch và tiến độ](docs/IMPLEMENTATION_PLAN.md).

Nguồn: [bài báo](https://aclanthology.org/2021.naacl-main.272/),
[code tác giả](https://github.com/dinobby/ZS-BERT).
