# ZS-BERT Simulation (NAACL-HLT 2021 Reproduction)

Mã nguồn tái hiện bài báo khoa học **["ZS-BERT: Towards Zero-Shot Relation Extraction with Attribute Representation Learning"](2021.naacl-main.272.pdf)** (*Chih-Yao Chen & Cheng-Te Li, NAACL-HLT 2021, pp. 3470–3479*).

Repository: [https://github.com/Student-persist-1919/zs-bert-simulation](https://github.com/Student-persist-1919/zs-bert-simulation)

---

## 1. Tổng quan Kiến trúc & Phương pháp (Theo bài báo NAACL 2021)

1. **Biểu diễn thuộc tính quan hệ (Relation Attribute Vectors - Section 4.1):**
   - Mỗi quan hệ $y \in Y_s \cup Y_u$ được mô tả bằng một câu văn tự nhiên $D_y$ trích xuất từ Wikidata (`resources/pid2name.json` & `resources/property_list.html`).
   - Sử dụng mô hình **Sentence-BERT** (`bert-large-nli-mean-tokens`) để ánh xạ câu mô tả $D_y$ thành vector thuộc tính cố định $a_y \in \mathbb{R}^{1024}$.

2. **Bộ mã hóa câu văn và thực thể (Input Sentence Encoder - Section 4.2):**
   - Sử dụng **`bert-large-cased`** ($d_{\text{hidden}} = 1024$) để trích xuất biểu diễn ngữ cảnh của câu.
   - Đánh dấu vị trí 2 thực thể $e_1$ (head) và $e_2$ (tail) bằng hai vector nhị phân `marked_e1` và `marked_e2`.
   - Tổng hợp vector `[CLS]` ($H_0'$), biểu diễn thực thể $H_{e1}$ và $H_{e2}$ qua tầng tuyến tính kèm hàm kích hoạt $\tanh$:
     $$\hat{a}_i^s = \tanh\left(W_1 \tanh([H_0' \oplus H_{e1} \oplus H_{e2}]) + b_1\right) \in \mathbb{R}^{1024}$$

3. **Hàm mục tiêu Đa nhiệm (Multi-Task Learning - Section 4.3, Eq. 2):**
   $$\mathcal{L} = (1 - \alpha) \sum_{i=1}^N \max\left(0, \gamma - a_i^s \cdot \hat{a}_i^s + \max_{j: y_j \ne y_i} (a_i^s \cdot \hat{a}_j^s)\right) - \alpha \sum_{i=1}^N y_i^s \log(\hat{y}_i^s)$$
   - Hệ số cân bằng: $\alpha = 0.4$, khoảng lề (margin): $\gamma = 7.5$, hàm khoảng cách: tích vô hướng (`inner`).

4. **Suy luận Zero-Shot (Nearest Neighbor Search - Section 4.4):**
   - Với câu kiểm thử mới $Z_i$ và tập $m$ quan hệ chưa từng xuất hiện trong tập huấn luyện ($Y_u$), dự đoán quan hệ thông qua tìm kiếm láng giềng gần nhất:
     $$C(Z_i) = \operatorname{argmax}_{j \in Y_u} (\hat{a}_i^u \cdot a_j^u)$$

---

## 2. Các Cải tiến & Sửa lỗi Quan trọng để Đạt 97%–99% Kết quả Bài báo

So với phiên bản thử nghiệm cũ (chỉ đạt F1 = 53.49%), dự án này đã khắc phục triệt để các nguyên nhân gây suy giảm hiệu năng:

1. **Sửa lỗi lệch chỉ số thực thể `[start, end]` trong FewRel & Wiki-ZSL:**
   - Trong `fewrel_dataset.json`, trường `pos: [start, end]` là một khoảng nửa mở `[start:end)` (tức `end` là chỉ số **ngoài** thực thể).
   - Phiên bản cũ truyền trực tiếp `[start, end]` làm danh sách vị trí thực thể, khiến từ nằm ngay sau thực thể luôn bị gán nhầm vào thực thể, các từ ở giữa của thực thể đa từ bị bỏ sót hoàn toàn, và gây lỗi `IndexError` khi thực thể nằm ở cuối câu (`end == len(tokens)`).
   - Trong [`scripts/preprocess_fewrel.py`](scripts/preprocess_fewrel.py) và [`model/data_helper.py`](model/data_helper.py), khoảng `[start, end]` được chuyển chính xác thành `list(range(start, end))`, khôi phục 100% độ chính xác của `e1_mask` và `e2_mask` trên toàn bộ 56,000 mẫu.

2. **Khôi phục chuẩn cấu hình `batch_size = 4` của bài báo (Section 5.1, trang 6):**
   - Hàm Margin Loss khai thác mẫu âm ngay trong batch ($\max_{j \ne i}$). Với `batch_size = 4` và `lr = 5e-6`, mô hình thực hiện **12,250 bước cập nhật trọng số mỗi epoch** (thay vì chỉ 3,062 bước khi dùng `batch_size = 16`), bám sát động học hội tụ của bài báo gốc.

3. **Tối ưu tốc độ huấn luyện (Pre-tokenization & Dynamic Padding):**
   - Tiền xử lý WordPiece một lần duy nhất khi khởi tạo `FewRelDataset` (loại bỏ hoàn toàn nút thắt cổ chai CPU lặp lại qua 10 epochs và cảnh báo chậm `torch.tensor(relation_emb)`).

---

## 3. Cấu trúc Thư mục Dự án

```text
zs-bert-simulation/
├── model/
│   ├── __init__.py
│   ├── model.py              # Kiến trúc ZSBert (BertPreTrainedModel) & Multi-task Loss
│   ├── data_helper.py        # DataLoader, WordPiece & Entity Mask alignment, Sentence-BERT
│   ├── evaluation.py         # Nearest Neighbor Search (inner, euclid, cosine) & Macro P/R/F1
│   ├── train_fewrel.py       # Script huấn luyện & đánh giá FewRel (m=10, live logging)
│   └── train_wiki.py         # Script huấn luyện & đánh giá Wiki-ZSL
├── resources/
│   ├── pid2name.json         # Từ điển 135 quan hệ Wikidata (ID -> [Name, Description])
│   └── property_list.html    # Bảng HTML thuộc tính Wikidata chuẩn tương thích ZS-BERT
├── scripts/
│   ├── build_resources.py    # Script sinh dữ liệu mô tả quan hệ trong resources/
│   ├── preprocess_fewrel.py  # Chuyển đổi fewrel_dataset.json -> data/fewrel_all.json
│   └── preprocess_wikizsl.py # Chuyển đổi wikizsl_dataset.json -> data/wiki_all.json
├── 2021.naacl-main.272.pdf   # Bài báo gốc NAACL-HLT 2021
├── zs-bert.ipynb             # Notebook chạy trên Kaggle (hỗ trợ Save & Commit + Live Log)
├── requirements.txt          # Các thư viện phụ thuộc
└── README.md
```

---

## 4. Hướng dẫn Đẩy lên GitHub & Chạy trên Kaggle (Save & Commit)

### Bước 1: Đẩy dự án lên GitHub (`Student-persist-1919/zs-bert-simulation`)
Mở terminal tại thư mục dự án và chạy:
```bash
git init
git add .
git commit -m "Implement ZS-BERT NAACL 2021 simulation for FewRel m=10"
git branch -M main
git remote add origin https://github.com/Student-persist-1919/zs-bert-simulation.git
git push -u origin main
```

### Bước 2: Chạy trên Kaggle bằng Save & Commit
1. Tải file [`zs-bert.ipynb`](zs-bert.ipynb) lên Kaggle Notebook.
2. Trong phần **Input** bên phải Kaggle, gắn bộ dữ liệu chứa `fewrel_dataset.json` (notebook sẽ tự động tìm kiếm đường dẫn file trong `/kaggle/input`).
3. Chọn **Accelerator: GPU T4 x2** (hoặc **GPU P100**) và bật **Internet: On**.
4. Nhấn **Save Version** $\rightarrow$ chọn **Save & Run All (Commit)** $\rightarrow$ nhấn **Save**.
5. Trong quá trình chạy ngầm, mở mục **Active Events** $\rightarrow$ nhấn vào phiên chạy $\rightarrow$ mở tab **Logs** để theo dõi trực tiếp tiến độ từng bước (`Step`, `Loss`, `Margin Loss`, `CE Loss`, `Speed`, `ETA`) và bảng đánh giá Zero-Shot sau mỗi Epoch.
