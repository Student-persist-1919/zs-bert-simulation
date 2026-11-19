# Chạy 3 cấu hình FewRel trên Kaggle

## Chuẩn bị dữ liệu

Tạo hoặc dùng một Kaggle Dataset có `fewrel_dataset.json` hoặc `fewrel_all.json`, sau đó Add Input vào notebook. Dữ liệu chuẩn có 80 quan hệ, mỗi quan hệ 700 mẫu. Dữ liệu Wiki-ZSL local phục vụ script cũ; ba notebook mới chỉ chạy FewRel.

Các file local được đặt tại `data/raw/` và không đưa lên GitHub. Notebook chỉ tìm dữ liệu trong `/kaggle/input`. Nếu có nhiều file trùng tên, gán `DATA_FILE` trong cell cấu hình tới đúng file.

Định dạng thực thể được hỗ trợ:

- FewRel gốc: `[name, id, [[token_indices]]]`; giữ nguyên danh sách chỉ số.
- Dạng dict: `{name, id, pos: [start, end]}`; mặc định khoảng nửa mở `[start, end)`, giống định dạng notebook đầu vào.
- Nếu dict.pos chứa danh sách chỉ số thay vì khoảng, đặt `DICT_POSITION_FORMAT = "indices"`.

Notebook dừng khi dữ liệu không đủ 56.000 mẫu, entity sai chỉ số hoặc thiếu mô tả quan hệ; không bỏ mẫu lỗi để tiếp tục.

## Import và chạy

1. Tải một file `.ipynb` trong `notebooks/kaggle/` từ GitHub; dùng chức năng import/upload notebook của Kaggle.
2. Bật GPU và Internet, Add Input FewRel, rồi chạy toàn bộ các cell.
3. Lặp lại cho hai file còn lại. Ba notebook độc lập, chỉ khác m và mốc đối chiếu bài báo.

Mỗi notebook chạy 5 seed tuần tự trong process riêng. Dùng GPU đầu tiên kể cả khi có T4 x2; batch train luôn là 4. Không tự chuyển sang huấn luyện đa GPU vì sẽ thay đổi các mẫu âm trong batch.

Cell đầu tự cài `transformers==4.57.3`, `sentence-transformers==5.2.0` và `tokenizers==0.22.2`; PyTorch của Kaggle cần >=2.4. Bộ ghim này có wheel tokenizers dùng được với Python 3.13. Notebook yêu cầu cài tokenizers từ wheel để tránh build Rust từ source; đồng thời kiểm tra import NLP trong process mới trước khi tải model.

Nếu đã chạy bản notebook cũ ghim transformers 4.44.2, hãy import lại file notebook đã sửa, restart session rồi chạy lại. Cảnh báo debugger về frozen modules không phải nguyên nhân lỗi cài dependency. Bộ dependency cũ kéo tokenizers 0.19.x, có thể khiến pip chuyển sang build source và thất bại trên Python 3.13.

## Kết quả

Thư mục `/kaggle/working/results_m_<m>/` chứa:

```text
results_m_<m>/
├── experiment_config.json       # Cấu hình, phiên bản, checksum và config hash
├── relation_descriptions.json   # Đúng các mô tả đã dùng
├── attributes.npz               # Các vector SBERT cố định, dùng cả khi resume
├── run_seed_300.json            # Tương tự cho 301–304
├── history_seed_300.json        # Kết quả qua từng epoch
├── train_seed_300.log           # Log riêng từng run
├── per_run_metrics.csv          # P/R/F1 từng run, best/final
├── aggregate_metrics.csv        # Trung bình, sample std và chênh lệch với báo
├── five_run_summary.json        # Tổng hợp đủ 5 runs
└── active_checkpoint.pt         # Chỉ tồn tại khi một run chưa hoàn tất
```

`five_run_summary.json` và CSV tổng hợp chỉ được tạo khi đủ 5 runs hoàn tất. Giá trị trong CSV tổng hợp là phần trăm; chênh lệch so với báo là điểm phần trăm. Trong JSON từng run, P/R/F1 lưu theo thang 0–1.

## Tiếp tục sau khi ngắt phiên

5 runs × 10 epoch có thể cần nhiều phiên GPU. Notebook lưu checkpoint mỗi 1.000 batch và sau mỗi epoch; chỉ giữ checkpoint đang hoạt động, xóa khi run hoàn tất. Checkpoint chứa trọng số, Adam, RNG và vị trí batch.

1. Trước khi mất session, lưu/tải toàn bộ thư mục `results_m_<m>/` làm outputs. Kaggle không bảo đảm khôi phục file chưa được lưu khi phiên bị ngắt.
2. Tạo Kaggle Dataset từ outputs đã lưu, Add Input vào phiên notebook tiếp theo.
3. Trong cell cấu hình đặt, ví dụ:

```python
RESUME_DIR = "/kaggle/input/my-previous-output/results_m_10"
```

4. Giữ nguyên m, seeds, dữ liệu, hyperparameter, runtime và loại GPU. Chạy lại toàn bộ notebook.

Notebook kiểm tra config hash, bỏ qua các run đã hoàn tất và tiếp tục run còn dang dở từ checkpoint gần nhất. Không nạp checkpoint từ nguồn không tin cậy: checkpoint dùng định dạng PyTorch có thể chứa đối tượng Python.

Giữ cả `experiment_config.json`, các `run_seed_*.json` đã hoàn tất, `attributes.npz` và `active_checkpoint.pt`; chỉ upload checkpoint riêng sẽ không đủ để tiếp tục chuỗi runs đúng trạng thái.
