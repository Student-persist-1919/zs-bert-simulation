# Hồ sơ thí nghiệm

- runs/: bằng chứng từng run chính thức do scripts/collect_results.py nhập từ ZIP.
- executed/: notebook có output, đặt tên gồm dataset/m/seed/attempt.
- validation/: ZIP và ghi nhận smoke test FewRel, Wiki-ZSL đã hoàn thành.
- historical/: số liệu trích log và các lần thử lỗi trước đây, không gộp vào kết quả mới.

Kết quả chung đặt cạnh ba notebook ở ../notebooks/kaggle/results.csv và RESULTS.md.
Checkpoint lớn giữ trên Kaggle, không commit vào GitHub.
Theo yêu cầu người dùng, bỏ qua bước kiểm thử pause/resume riêng để chạy chính thức.
Cơ chế checkpoint/deadline vẫn có trong runner; GPU resume chưa được kiểm chứng.
