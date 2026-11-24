# Lưu ý về phiên bản tài liệu

Phần bên dưới mô tả **bộ notebook FewRel năm seed cũ**, lưu để truy nguyên.
Bộ hiện tại dùng 3 notebook một seed/hai dataset, clone CODE_REV do người dùng
chọn rồi ghi commit thực tế, không còn tự khóa upstream 7a37cf5.
Hướng dẫn hiện hành: [KAGGLE_SINGLE_RUN.md](KAGGLE_SINGLE_RUN.md).
Wiki-ZSL local 93.383/113 khác số mẫu báo; chưa xác minh nguồn tác giả end-to-end.
Bộ refactor hiện đã qua smoke GPU cả hai dataset và có full runs FewRel.
Ngày 2026-11-24 đã nhập 6 run complete đủ 10 epoch; giữ riêng các attempt paused.
Resume thực tế trên T4 đã hoàn tất ở ranh giới epoch và giữa epoch; chưa kiểm tra
tương đương từng bit với chạy liên tục. Xem [hồ sơ hiện tại](../experiments/README.md)
và [notebook cùng link Kaggle](../experiments/executed/README.md).
Các câu về trạng thái chưa chạy GPU ở phần lịch sử bên dưới là ghi nhận trước đây.

---

# Thiết lập và giới hạn tái lập

## Phạm vi

Ba notebook trong `notebooks/kaggle/` chạy FewRel với m=5, m=10, m=15. Mỗi m có 5 lần chọn quan hệ unseen, dùng seed 300, 301, 302, 303, 304. Mỗi lần bắt đầu lại model và Adam; SBERT cung cấp vector mô tả cố định.

FewRel cần đủ 80 quan hệ × 700 mẫu. Toàn bộ mẫu của các quan hệ seen dùng để train, toàn bộ mẫu của m quan hệ unseen dùng để test, theo cách chia trong script tác giả. Với m=5/10/15, số mẫu train lần lượt là 52.500/49.000/45.500; số mẫu test là 3.500/7.000/10.500.

## Nguồn được cố định

- Implementation upstream: `Student-persist-1919/zs-bert-simulation`, commit `7a37cf51435b8ab8726d74ad1f9f374fbbae88b0`.
- Bảng mô tả quan hệ: repository tác giả `dinobby/ZS-BERT`, commit `e2ffaea66244b2905055ee7cc47917985c3eab60`; SHA256 `d9c91ac01f18398e58a48ad7913e3f168553e3c06b306879b1b2389bde3eb21d`.
- `google-bert/bert-large-cased`: revision `06fa25dc1b6e034e91607fd67c247051ccaeff65`.
- `sentence-transformers/bert-large-nli-mean-tokens`: revision `a85b0d10e16867eb768e411c66f79475505b65dd`.
- Thư viện chính: `transformers==4.57.3`, `sentence-transformers==5.2.0`, `tokenizers==0.22.2`; phiên bản torch/numpy/scikit-learn thực tế được ghi trong outputs. Bộ ghim được cập nhật để hỗ trợ môi trường Kaggle Python 3.13, tránh lỗi build tokenizers của bộ dependency cũ. Wheel Linux cho Python 3.13 đã được kiểm tra tải thành công; chưa chạy end-to-end trên GPU Kaggle.

Tài nguyên `resources/property_list.html` của implementation cũ được giữ nguyên để bảo toàn các script cũ. Ba notebook mới tải snapshot bảng mô tả của tác giả; hai bảng này có một số mô tả khác nhau.

## Tham số

Adam; lr=5e-6; batch train=4; gamma=7.5; alpha=0.4; negative inner product / exact argmax inner product khi dự đoán. BERT-large-cased; vector câu và vector quan hệ 1024 chiều. SBERT không được cập nhật trong lúc train.

Chạy FP32, attention eager, không dùng AMP, không nhân batch theo số GPU và không tích lũy gradient. Gradient checkpointing giảm VRAM. Thứ tự nhãn và permutation của từng epoch được xác định để có thể tiếp tục tại batch checkpoint.

Hai điểm không được báo mô tả đủ để hứa tái lập nguyên xi:

- Báo §5.1 nói lặp lại 5 lần nhưng không công bố bộ 5 seed; 300–304 là lựa chọn tái lập.
- Báo ghi hidden size 768, nhưng code tác giả dùng BERT-large hidden 1024. Notebook theo code tác giả và notebook đầu vào. 10 epoch cũng lấy từ README/script tác giả, không phải giá trị nêu trong §5.1.

Implementation giữ cách đánh dấu entity theo token gốc +1; không căn chỉnh lại entity sau khi WordPiece tách từ. Đây là giới hạn kế thừa cần ghi nhận khi diễn giải kết quả, không phải khẳng định mask entity đã hoàn toàn chính xác.

## Metric và cách tổng hợp

Kết quả chính gọi hàm `compute_macro_PRF` của tác giả: precision chia cho số nhãn được dự đoán, recall chia cho số nhãn thật, rồi F1 là harmonic mean của hai đại lượng này. Hàm đó khác `sklearn` macro-F1 chuẩn. Outputs bổ sung `standard_macro_precision`, `standard_macro_recall`, `standard_macro_f1` trên đầy đủ m nhãn để kiểm tra.

`best` lấy epoch có test F1 cao nhất trong mỗi run, tương ứng cách script tác giả theo dõi best. Việc chọn bằng unseen test có thể làm kết quả lạc quan. `final` là epoch 10 cố định, được báo cáo riêng. Không lấy best run trong 5 runs; mỗi metric được lấy trung bình trên cả 5 runs, độ lệch chuẩn mẫu ddof=1. F1 trung bình được tính từ 5 F1, không tính lại từ P/R trung bình.

Mốc P/R/F1 trong Table 2 của báo, đơn vị phần trăm:

- m=5: 76.96 / 78.86 / 77.90.
- m=10: 56.92 / 57.59 / 57.25.
- m=15: 35.54 / 38.19 / 36.82.

Notebook dùng exhaustive argmax thay BallTree vì negative inner product không phải metric phù hợp cho pruning của BallTree; thứ tự nhãn cũng được xác định thay vì phụ thuộc thứ tự `set`.

## Trạng thái kiểm tra

Đã kiểm tra cấu trúc notebook, compile các cell Python và script nhúng, xử lý entity, số lượng mẫu/split và cell tổng hợp với dữ liệu kiểm thử tổng hợp. Chưa huấn luyện trên GPU Kaggle; chưa có kết quả thực nghiệm mới để khẳng định bằng điểm của báo.

Notebook cũ và README cũ được giữ dưới `notebooks/archive/` và `docs/legacy/`. Các tuyên bố tỷ lệ tái lập trong README cũ là nội dung lịch sử, không đại diện cho kết quả đã được xác minh của bộ 5 runs mới.
