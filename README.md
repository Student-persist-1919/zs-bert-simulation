# ZS-BERT — FewRel: m=5, 10, 15 / 5 runs

Dự án tái lập thí nghiệm zero-shot relation extraction của bài báo **ZS-BERT: Towards Zero-Shot Relation Extraction with Attribute Representation Learning** (NAACL 2021).

Ba notebook bên dưới có thể import riêng vào Kaggle và chạy độc lập. Mỗi notebook chạy **5 seed: 300–304**, chọn lại tập quan hệ unseen, khởi tạo model/optimizer mới và xuất **P/R/F1 trung bình ± độ lệch chuẩn mẫu**.

- [FewRel m=5 — 5 runs](notebooks/kaggle/zs-bert-m-5-5runs.ipynb)
- [FewRel m=10 — 5 runs](notebooks/kaggle/zs-bert-m-10-5runs.ipynb)
- [FewRel m=15 — 5 runs](notebooks/kaggle/zs-bert-m-15-5runs.ipynb)

## Cấu trúc thư mục

```text
BTL/
├── notebooks/
│   ├── kaggle/
│   │   ├── zs-bert-m-5-5runs.ipynb
│   │   ├── zs-bert-m-10-5runs.ipynb
│   │   └── zs-bert-m-15-5runs.ipynb
│   └── legacy/
│       └── zs-bert.ipynb              # Notebook cũ, chạy một seed
├── model/
│   ├── __init__.py
│   ├── model.py
│   ├── data_helper.py
│   ├── evaluation.py
│   ├── train_fewrel.py               # Script một run cũ
│   └── train_wiki.py
├── scripts/
│   ├── build_resources.py
│   ├── preprocess_fewrel.py          # Script tiền xử lý cũ
│   └── preprocess_wikizsl.py
├── resources/
│   ├── pid2name.json
│   └── property_list.html           # Tài nguyên của implementation cũ
├── docs/
│   ├── KAGGLE.md                    # Chạy và tiếp tục sau khi ngắt phiên
│   ├── REPRODUCIBILITY.md           # Cấu hình, metric và giới hạn tái lập
│   ├── papers/
│   │   └── 2021.naacl-main.272.pdf
│   ├── legacy/
│   │   └── README-original.md       # Tài liệu phiên bản cũ, lưu tham khảo
│   └── local/                      # Tài liệu cá nhân; không đưa lên Git
├── data/
│   └── raw/                        # Dữ liệu local; không đưa lên Git
│       ├── fewrel_dataset.json
│       └── wikizsl_dataset.json
├── requirements.txt                # Phụ thuộc của implementation cũ
├── requirements-kaggle.txt          # Phụ thuộc tương ứng 3 notebook mới
├── .gitignore
└── README.md
```

`data/` và `docs/local/` chỉ có trên máy local, được bỏ qua bởi Git. Thư mục kết quả/checkpoint do Kaggle tạo lúc chạy cũng không được đưa vào repo.

## Chạy trên Kaggle

1. Mở một notebook trong `notebooks/kaggle/`, tải file `.ipynb` từ GitHub và import vào Kaggle.
2. Bật **GPU** và **Internet** cho notebook.
3. Add Input bộ FewRel chứa `fewrel_dataset.json` hoặc `fewrel_all.json`: đủ **80 quan hệ × 700 mẫu = 56.000 mẫu**. Có thể upload file local từ `data/raw/` lên một Kaggle Dataset riêng.
4. Chạy **Run All** hoặc lưu một phiên chạy toàn bộ notebook.
5. Tải kết quả từ `/kaggle/working/results_m_5/`, `results_m_10/` hoặc `results_m_15/`.

Mỗi notebook tự cài các thư viện cần thiết, tải code/model và chứa script điều phối 5 runs. Không cần upload thư mục `model/`, `scripts/` hoặc PDF lên Kaggle. Chi tiết: [hướng dẫn Kaggle](docs/KAGGLE.md).

## Thiết lập thí nghiệm

- Train chỉ dùng các quan hệ seen; test chỉ dùng m quan hệ unseen, hai tập nhãn rời nhau.
- Mỗi m chạy 5 lần với seed khác nhau; không chọn run tốt nhất để đại diện cho cả thí nghiệm.
- Adam; learning rate `5e-6`; batch train `4`; gamma `7.5`; alpha `0.4`; inner product.
- BERT-large-cased và Sentence-BERT bert-large-nli-mean-tokens; vector quan hệ 1024 chiều, SBERT cố định.
- 10 epoch theo lệnh chạy của code tác giả; FP32; gradient checkpointing để giảm VRAM.
- Kết quả có cả **best epoch của từng run** theo cách script tác giả theo dõi, và **epoch cuối** để đối chiếu. `best` có sử dụng test F1 để chọn epoch.

Báo không công bố bộ 5 seed gốc. Seed 300–304 là lựa chọn tái lập; không bảo đảm khớp điểm công bố. Có khác biệt giữa mô tả hidden size trong báo và BERT-large trong code tác giả. Xem [chi tiết tái lập](docs/REPRODUCIBILITY.md).

## Đưa thay đổi lên GitHub

Repo local đã có remote `origin`; chạy trong thư mục `BTL`:

```powershell
git status --short
git diff --stat
git add README.md .gitignore requirements-kaggle.txt notebooks docs
git add -u -- zs-bert.ipynb 2021.naacl-main.272.pdf
git diff --cached --stat
git commit -m "Organize Kaggle notebooks for FewRel m=5,10,15 with five runs"
git push origin main
```

Hai lệnh `git add` ghi nhận cả file mới và việc chuyển notebook/PDF cũ. Dataset, file cá nhân và checkpoint bị `.gitignore` loại khỏi staging. Kiểm tra `git diff --cached --stat` trước khi commit.

Các notebook cố định code upstream tại commit `7a37cf51435b8ab8726d74ad1f9f374fbbae88b0`, thay vì clone phiên bản mới nhất. Vì vậy việc sắp xếp lại repo ở commit mới không làm thay đổi code dùng cho thí nghiệm. Muốn thay model/loss cho một thí nghiệm mới, cần cập nhật notebook và revision có chủ đích.

## Tham khảo

- [Bài báo trên ACL Anthology](https://aclanthology.org/2021.naacl-main.272/)
- [PDF trong repo](docs/papers/2021.naacl-main.272.pdf)
- [Mã nguồn tác giả](https://github.com/dinobby/ZS-BERT)
- [Repository implementation](https://github.com/Student-persist-1919/zs-bert-simulation)
