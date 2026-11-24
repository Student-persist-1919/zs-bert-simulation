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

## Bằng chứng đã nhập ngày 2026-11-24

- FewRel m=5: seeds 300, 301, 302 complete. Seed 301 dùng cặp chạy lại/resume trên ncmanh; bản paused cũ được giữ.
- FewRel m=10: seeds 300, 301, 302 complete.
- Chưa nhóm nào đủ seeds 300–304, nên chưa tính trung bình 5 runs.
- Wiki-ZSL và m=15 chưa có full run được nhập; smoke test được giữ riêng.

Kiểm tra lại ngày 2026-11-24: đủ run 1–3 (seeds 300–302) cho từng m=5 và m=10.
Cả 6 kết quả complete đều đủ epoch 1–10; best/final, split seen/unseen và CSV khớp
bằng chứng. Đã đối chiếu CRC/SHA256 của 8 ZIP attempt gốc với file đã nhập và
SHA256 của 12 notebook chính thức có output; CONFIG/STATUS và các dòng EVALUATION
khớp bằng chứng của phiên tương ứng, không có output lỗi thực thi.
Cặp seed 301 m=5 chạy lại/resume giữ nguyên prefix history và provenance.
Hai attempt paused không được đếm thành run complete; mỗi m vẫn còn thiếu run 4–5.

[Mục lục notebook và link từng phiên Kaggle](executed/README.md).
Mỗi thư mục `runs/<dataset>_m<m>_seed<seed>/original/` giữ nguyên bằng chứng
đã giải nén, `import_record.json` ghi link version và SHA256 ZIP;
`validation_record.json` ghi các bước đối chiếu và SHA256 notebook.
Với run đã resume, notebook paused và complete được giữ riêng; ZIP cuối chứa
toàn bộ history. Attempt paused chưa hoàn thành vẫn được lưu, không tính vào mean.

Thư mục gom `submissions_local/` chỉ dùng local và được `.gitignore` bỏ qua.
Các file đã kiểm tra được sao chép sang `experiments/` để giáo viên theo dõi.
Giữ sẵn khung `submissions_local/<dataset>/m<m>/run_<id>/` cho cả 30 run:
FewRel/Wiki-ZSL × m=5/10/15 × run 1–5, kể cả thư mục đang chờ kết quả.
Mỗi run có mẫu `links.txt` để lưu nguồn và ghi chú; giữ các file nhận xét, kiểm tra
để tổng hợp sau cùng. Giữ ZIP gốc để đối chiếu nguồn và checksum, các attempt
paused, notebook cũ và hồ sơ historical/archive để truy nguyên. Các notebook
`notebooks/archive/five-runs/` cho m=5/10/15 cũng được giữ nguyên.
Khi dọn dự án, giữ thư mục phục vụ kết quả, nhận xét và tổng hợp dù chưa có kết quả;
cache `__pycache__/` có thể xóa và được Python tạo lại khi cần.
