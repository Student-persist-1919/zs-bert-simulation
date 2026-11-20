# Kế hoạch tái lập ZS-BERT trên Kaggle Free

Phạm vi: tái lập ZS-BERT trong Table 2, hai dataset FewRel/Wiki-ZSL,
m=5/10/15, mỗi cấu hình 5 seed 300–304, mỗi run 10 epoch.
30 runs hoàn chỉnh có thể cần hơn 30 phiên Save & Run All (giới hạn 12 giờ).

## Các bước thực hiện

1. Bảo toàn notebook cũ và bằng chứng hai seed 300; tách notebook điều phối
   khỏi script train, chuẩn hóa, tải dữ liệu và tổng hợp dùng chung.
2. Kiểm tra nguồn dữ liệu, checksum, entity spans, mô tả quan hệ. FewRel local
   có 56.000 mẫu/80 quan hệ; Wiki-ZSL local 93.383/113, khác Table 1 (94.383).
   Không tự thêm/xóa mẫu để ép khớp báo. Nguồn mặc định là link tác giả.
3. Tạo 3 notebook m=5/10/15. Chọn DATASET và RUN_ID, tự tải code từ GitHub,
   dữ liệu từ nguồn tác giả, model revision cố định. Ghi commit thực tế.
4. Mỗi phiên chỉ train một seed. Giữ FP32, Adam, batch 4, lr 5e-6,
   gamma 7.5, alpha 0.4. Deadline tính từ cell đầu, lưu checkpoint và kết thúc
   bình thường trước 12 giờ; resume cùng seed khi cần. Chỉ complete sau 10 epoch.
5. Xuất history, metrics best/final, standard macro metrics, split, config,
   runtime, checksum và log. ZIP bằng chứng nhẹ không chứa checkpoint/model.
6. Nhập bằng chứng bằng script local; results.csv và RESULTS.md cùng cấp
   ba notebook. Kiểm tra seed/config/10 epoch; tổng hợp mean và sample std
   khi đủ 5 seed. Giữ executed notebooks tải từ đúng Kaggle version.
7. Kiểm tra local: normalize cả hai dataset, compile notebook/scripts,
   kiểm tra nhập/tổng hợp và checkpoint/deadline qua thử nghiệm nhỏ.
8. Sau khi push code: smoke test GPU Kaggle cả hai dataset, kiểm tra pause/resume,
   đo thời gian rồi cố định CODE_REV cho chuỗi runs chính thức. Không dùng smoke
   test làm kết quả chính thức. Lưu bằng chứng sau từng run và cập nhật báo cáo.

## Thư viện và tính trung thực

Lưu requirements tác giả để tham khảo; dùng requirements Kaggle tương thích
Python/CUDA hiện hành và lưu môi trường thực tế. Không tuyên bố giữ nguyên
runtime 2021. Ghi nhận hidden size trong báo khác implementation, entity mask
theo token gốc, metric tác giả khác sklearn, và best chọn bằng unseen test F1.
Không gộp kết quả khác cấu hình. Seed 300–304 là lựa chọn tái lập.

## Trạng thái

- Đã xác minh main GitHub và local cùng commit 122aa567 khi bắt đầu.
- Hai run FewRel seed 300 hoàn thành theo log; bản JSON trích log là bằng chứng
  lịch sử, chưa tự động nhập vào chuỗi thí nghiệm mới.
- Chưa chạy bộ code refactor trên GPU Kaggle; chưa hoàn thành 30 runs.
- Điểm cần xác minh ngoài local: dataset tải từ link tác giả, khác biệt Wiki-ZSL,
  môi trường GPU, thời gian một run và resume thực tế.

## Đã triển khai trong đợt refactor

- Bước 1: chuyển 3 notebook năm seed sang notebooks/legacy/five-runs;
  JSON trích log sang notebooks/kaggle/historical, giữ nguyên nội dung.
- Bước 2: đếm/chuẩn hóa local thành công cả 2 dataset; ghi checksum thực tế lúc
  tải, kiểm tra entity/count, dừng khi Wiki-ZSL lệch số mẫu chưa được xác nhận.
- Bước 3–6: đã tạo notebook, runner chung, deadline/checkpoint, ZIP bằng chứng,
  importer kiểm tra hash và bộ tổng hợp đúng 5 seed; hướng dẫn lưu executed notebook.
- Bước 7: compile scripts/cells, normalize dữ liệu local và unit tests đạt.
  Deadline helper được kiểm tra bằng đồng hồ giả; chưa kiểm tra PyTorch checkpoint
  hoặc GPU resume thực tế. Local không có PyTorch, nên không tuyên bố train đã đạt.
- Bước 8 còn lại: push bộ code, kiểm tra tải dữ liệu thực từ nguồn tác giả,
  smoke GPU hai dataset và pause/resume trên Kaggle, chốt revision/hash,
  chạy 30 runs và nhập kết quả thực. results.csv hiện chỉ có header.

## Sửa sau smoke test Kaggle đầu tiên

- Log người dùng: Python 3.13/PyTorch 2.11/Tesla T4 import đạt, lỗi gdown public
  file retrieval sau khoảng 60 giây, chưa train. Chưa xác định quota hay quyền Drive.
- Test tái hiện đường chạy main khi Drive unavailable đã thất bại trước bản sửa.
- Thêm resources/datasets/*.json.gz giữ nguyên byte dữ liệu BTL, SHA256 gzip/raw,
  thống kê và nguồn; mặc định snapshot, author-drive là tùy chọn rõ ràng.
- Không tuyên bố snapshot đã xác minh giống dữ liệu tác giả. Chặn count Wiki-ZSL
  chưa được cho phép như trước. Notebook không cần dataset Input cho run mới.
- Cần người dùng chạy lại smoke test Kaggle; test local chỉ xác nhận chuẩn bị dữ liệu.

## Kết quả smoke test GPU sau bản sửa

FewRel m=10 seed 300 tại commit b4b8420 đã hoàn thành 1 epoch/280 train/40 test
trên Tesla T4, xuất evidence ZIP hợp lệ. Bằng chứng và kiểm tra nằm trong
notebooks/kaggle/validation/. Chưa tính vào kết quả chính thức. Còn cần notebook
đã chạy, smoke Wiki-ZSL, GPU pause/resume và bộ 30 runs đủ 10 epoch.
