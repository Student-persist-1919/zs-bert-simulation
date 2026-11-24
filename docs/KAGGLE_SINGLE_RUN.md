# Chạy từng run trên Kaggle Free

## Chạy ngay từng run đủ 10 epoch

FewRel và Wiki-ZSL đã qua smoke test. Người dùng quyết định bỏ qua kiểm thử
pause/resume riêng; không có bước thử bổ sung bắt buộc trước khi chạy chính thức.

1. Import notebooks/kaggle/zs-bert-m-{5,10,15}.ipynb vào Kaggle mới.
2. GPU On, Internet On. Không cần dataset Input khi bắt đầu run mới.
3. Chọn DATASET="fewrel" hoặc "wikizsl" và RUN_ID=1..5.
4. Giữ SMOKE_TEST=False, CODE_REV full SHA b4b84208cf43b4274f2d2aaaf14beead0daf4707.
   Đây là runner đã chạy smoke GPU; notebook bootstrap mới checkout được SHA này.
   EXPECTED_DATA_SHA256 tự chọn theo dataset, không cần sửa. M đã cố định theo tên file.
5. Save Version → Save & Run All. Kiểm tra log in epochs=10, smoke=false.
6. Khi complete: tải ZIP evidence và notebook có output, gửi để nhập vào hồ sơ GitHub.
7. Tăng RUN_ID, tạo version/session mới; sau 5 run đổi dataset. Lặp cho cả 3 m.
   Cần 30 runs hoàn chỉnh; một run paused chưa tính hoàn thành.

Kaggle báo phiên `successful` chỉ nghĩa là notebook kết thúc không lỗi.
Luôn kiểm tra `status.json`: chỉ `status="complete"` và history đủ epochs 1–10
mới tính là một run hoàn thành; `paused` vẫn phải tiếp tục cùng RUN_ID.

Tệp kết quả cũ trích log không được tự gộp; các run chính thức mới có hash riêng.
Nếu sửa cell cấu hình, sửa giá trị tại dòng hiện có; không dán thêm biến trùng tên.

## Dữ liệu

Mặc định DATA_SOURCE="snapshot": clone bản JSON nén từ resources/datasets cùng
commit code, kiểm tra SHA256 của cả archive và dữ liệu giải nén, rồi kiểm tra
JSON/entities/counts. Đây là file gốc trong BTL/data/raw; chưa xác nhận giống hệt
file Google Drive tác giả. DATA_SOURCE="author-drive" là lựa chọn rõ ràng khi muốn
kiểm tra link tác giả; không tự thay nguồn nếu tải lỗi.
Nguồn: https://github.com/dinobby/ZS-BERT#dataset
FewRel phải 80 quan hệ × 700 mẫu. Wiki-ZSL phải 113 quan hệ; Table 1 ghi 94.383 mẫu.
Snapshot Wiki-ZSL có 93.383 mẫu. Nếu bản tải tác giả cũng khác báo, phải xem nguồn và
ghi giải trình. Bộ notebook chính đặt ALLOW_WIKI_COUNT_MISMATCH theo DATASET để
cho phép chạy snapshot hiện có, vẫn ghi discrepancy vào provenance. Không tự sửa số mẫu.
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
version. Lưu notebook vào experiments/executed với tên gồm dataset/m/seed/attempt.
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
Đã có smoke GPU cả hai dataset. Ngày 2026-11-24, đã nhập 6 full run FewRel
complete; các attempt paused được giữ riêng; xem [hồ sơ hiện tại](../experiments/README.md).
Resume ở đầu epoch và giữa epoch đã hoàn tất trên T4 trong các run chính thức.
Không suy ra tương đương từng bit với chạy liên tục từ kiểm tra này.
