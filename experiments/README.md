# Hồ sơ thí nghiệm

- runs/: bằng chứng từng run chính thức do scripts/collect_results.py nhập từ ZIP.
- executed/: notebook có output, đặt tên gồm dataset/m/seed/attempt.
- validation/: ZIP và ghi nhận smoke test FewRel, Wiki-ZSL đã hoàn thành.
- historical/: số liệu trích log và các lần thử lỗi trước đây, không gộp vào kết quả mới.

Kết quả chung đặt cạnh ba notebook ở ../notebooks/kaggle/results.csv và RESULTS.md.
Checkpoint lớn giữ trên Kaggle, không commit vào GitHub.
Theo yêu cầu người dùng, bỏ qua bước kiểm thử pause/resume riêng để chạy chính thức.
Cơ chế checkpoint/deadline được giữ và đã quan sát resume thành công trên T4
trong các run chính thức. Chưa kiểm tra tương đương từng bit với chạy liên tục.

## Bằng chứng đã nhập ngày 2026-11-23

- FewRel m=5: seed 300 và 302 complete; seed 301 paused, chưa có điểm cuối.
- FewRel m=10: seeds 300, 301, 302 complete.
- Chưa nhóm nào đủ seeds 300–304, nên chưa tính trung bình 5 runs.
- Wiki-ZSL và m=15 chưa có full run được nhập; smoke test được giữ riêng.

[Mục lục notebook và link từng phiên Kaggle](executed/README.md).
Mỗi thư mục `runs/<dataset>_m<m>_seed<seed>/original/` giữ nguyên bằng chứng
đã giải nén, `import_record.json` ghi link version và SHA256 ZIP;
`validation_record.json` ghi các bước đối chiếu và SHA256 notebook.
Với run đã resume, notebook paused và complete được giữ riêng; ZIP cuối chứa
toàn bộ history. Attempt paused chưa hoàn thành vẫn được lưu, không tính vào mean.

Thư mục gom `submissions_local/` chỉ dùng local và được `.gitignore` bỏ qua.
Các file đã kiểm tra được sao chép sang `experiments/` để giáo viên theo dõi.
