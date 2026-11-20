# Chạy từng run trên Kaggle Free

## Chuẩn bị và thứ tự

1. Push bộ code refactor lên repo trước khi import notebook mới. Ba notebook mới
   cần scripts/kaggle_run.py có trong commit được tải. Chưa push thì không chạy được.
2. Import một trong notebooks/kaggle/zs-bert-m-{5,10,15}.ipynb.
3. GPU On, Internet On. Không cần dataset Input khi bắt đầu run mới.
4. Chọn DATASET và RUN_ID (1–5, seed 300–304). SMOKE_TEST mặc định True:
   một epoch, tối đa 4 mẫu mỗi quan hệ, chỉ kiểm tra kỹ thuật, không phải kết quả.
5. Save Version → Save & Run All. Kiểm tra log, output, đánh giá và trạng thái.
6. Kiểm tra cả hai dataset; đặt budget ngắn để kiểm tra pause rồi resume cùng seed.
7. Ghi SHA commit CODE COMMIT và raw_data_sha256; gán CODE_REV và
   EXPECTED_DATA_SHA256 tương ứng cho bộ runs chính thức. Đặt SMOKE_TEST=False.
   Ưu tiên SHA đầy đủ 40 ký tự. Bootstrap mới fetch refs rồi resolve SHA ở local;
   bootstrap cũ không hỗ trợ fetch SHA rút gọn. Khi sửa cell, giữ đủ M và DATA_FILE,
   mỗi biến chỉ gán một lần để không vô tình ghi đè lựa chọn trước đó.
8. Mỗi phiên chỉ chạy một run. Sau complete đủ 10 epoch mới tính là kết quả chính thức.

## Dữ liệu

Mặc định DATA_SOURCE="snapshot": clone bản JSON nén từ resources/datasets cùng
commit code, kiểm tra SHA256 của cả archive và dữ liệu giải nén, rồi kiểm tra
JSON/entities/counts. Đây là file gốc trong BTL/data/raw; chưa xác nhận giống hệt
file Google Drive tác giả. DATA_SOURCE="author-drive" là lựa chọn rõ ràng khi muốn
kiểm tra link tác giả; không tự thay nguồn nếu tải lỗi.
Nguồn: https://github.com/dinobby/ZS-BERT#dataset
FewRel phải 80 quan hệ × 700 mẫu. Wiki-ZSL phải 113 quan hệ; Table 1 ghi 94.383 mẫu.
Snapshot Wiki-ZSL có 93.383 mẫu. Nếu bản tải tác giả cũng khác báo, phải xem nguồn và
ghi giải trình; chỉ sau đó đặt ALLOW_WIKI_COUNT_MISMATCH=True. Không tự sửa số mẫu.
DATA_FILE là tùy chọn khi kiểm tra local hoặc tải tác giả bị gián đoạn.

Lần smoke test Kaggle đầu đã nhận GPU T4 nhưng dừng ở gdown: không lấy được link
FewRel của tác giả. Cảnh báo dependency Gradio/Diffusers không phải lỗi làm lần
chạy đó dừng. Snapshot được thêm sau lỗi này để loại phụ thuộc vào Google Drive.
Chưa xác minh end-to-end nguồn Drive; không coi snapshot là dữ liệu tác giả đã xác minh.

## Deadline và tiếp tục

MAX_SESSION_HOURS mặc định 10.5, tính từ cell đầu, tối đa cấu hình 11 giờ.
Trainer kiểm tra deadline giữa các batch và trước evaluation, lưu model/Adam/RNG/
vị trí batch, ghi paused, exit thành công. Checkpoint ghi tạm rồi thay thế nguyên tử.
Evaluation/download/init không có hard timeout theo budget: cần smoke test GPU để
xác nhận khoảng dự phòng đủ rộng. Không có cam kết tuyệt đối chống mọi lỗi nền tảng.

Nếu paused: giữ toàn bộ output trên Kaggle, gắn output vào phiên mới, đặt RESUME_DIR
tới thư mục run, giữ dataset/m/run-id/code/runtime/data. Bắt đầu lại toàn bộ notebook
để có deadline mới. Resume yêu cầu khớp config, GPU, packages, data và fixed attributes.
Checkpoint bộ notebook cũ không tự tương thích bộ mới. Một run có thể cần >1 phiên.

## Bằng chứng và kết quả GitHub

Download *_evidence.zip từ Output (không có checkpoint) và .ipynb đã chạy từ đúng
version. Lưu notebook vào notebooks/kaggle/executed với tên gồm dataset/m/seed/attempt.
Ví dụ lệnh local:

```powershell
python scripts/collect_results.py --zip D:/Downloads_D/fewrel_m5_seed300_evidence.zip --attempt original --kaggle-version-url https://www.kaggle.com/code/USER/SLUG?scriptVersionId=VERSION
```

Script kiểm tra manifest SHA256, cấu hình, status và history. SHA256 là kiểm tra tính
nhất quán, không phải chữ ký chứng minh nguồn. Link Kaggle là bằng chứng đối chiếu.
Lưu attempt bất biến; gọi lại cùng tên sẽ báo lỗi. results.csv và RESULTS.md được
tạo lại cùng cấp 3 notebook. Thiếu seed, duplicate attempts hoặc khác cấu hình thì
không tạo trung bình 5 runs. Cần quyết định/document attempt nào hợp lệ trước tổng hợp.

GitHub không tự nhận thay đổi file từ session Kaggle. Bước download/import/commit
thực hiện local; không đặt GitHub token trong notebook. Checkpoint giữ trên Kaggle.

## Tương thích và giới hạn

requirements-original.txt là bản tham khảo tác giả (2020). requirements-kaggle.txt
là bộ tương thích hiện đại, không thay PyTorch CUDA của Kaggle. Mỗi run ghi runtime.
Theo model/loss implementation hiện có: BERT-large 1024, FP32, batch 4, Adam,
lr 5e-6, gamma 7.5, alpha 0.4. Giữ entity mask token gốc+1; chưa sửa WordPiece
alignment. Điểm này khác một implementation căn chỉnh entity đầy đủ.
Metric chính theo tác giả; thêm sklearn macro để kiểm tra. Best dùng test F1 để chọn
epoch; final là epoch 10. Không tuyên bố trùng kết quả báo hay runtime 2021.

30 runs tương ứng 2 datasets × 3 m × 5 seeds; có thể cần hơn 30 Save Versions.
Chưa chạy full train hoặc smoke GPU cho bộ refactor; các bước này phải thực hiện
trên Kaggle và giữ bằng chứng trước khi công bố kết quả.
