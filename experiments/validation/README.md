# Kiểm thử Kaggle

Đây là bằng chứng kiểm thử kỹ thuật; không nhập vào results.csv chính thức.

## FewRel m=10, seed=300: smoke test đạt

- Evidence: [fewrel_m10_seed300_smoke_evidence.zip](fewrel_m10_seed300_smoke_evidence.zip).
- ZIP SHA256: `912c183bdc4513061fb1ec2f2ada1637699372d530f634c8cb8c0e8244abb385`.
- Code commit theo provenance: `b4b84208cf43b4274f2d2aaaf14beead0daf4707`.
- Kaggle version theo ảnh người dùng:
  https://www.kaggle.com/code/nqmoc1/zs-bert-m-10?scriptVersionId=371645302
- Python 3.13.16, PyTorch 2.11.0+cu130, Tesla T4.
- Snapshot FewRel raw hash:
  `85c2595bf743c203c22e78868f5a9f309ad726aa27c36e804554d99461736e19`.
- 280 train / 40 test, 70 seen / 10 unseen quan hệ, không trùng nhãn.
- SMOKE_TEST=True, 1 epoch, status complete. Không phải run chính thức 10 epoch.
- P/R/F1: 10.9259% / 15% / 12.6429%; không dùng so sánh chất lượng với báo.
- Đã kiểm tra CRC ZIP, toàn bộ SHA256 trong manifest, config hash khớp giữa
  config/result/status, history file khớp history trong result và log.
- Notebook có output đã nhận: [fewrel_m10_seed300_smoke.ipynb](../executed/fewrel_m10_seed300_smoke.ipynb).
  SHA256: `d718eb9121f8b54d061d4f318874f5b74942d43e73f322e197b2154f07256c34`.
  Config trong output khớp hoàn toàn config trong ZIP, Papermill exception=null,
  không có error output; code commit thực tế là b4b8420 dù biến CODE_REV còn main.
- Ở thời điểm lưu FewRel, chưa kiểm tra Wiki-ZSL hoặc GPU pause/resume; xem cập nhật bên dưới.

ZIP chứa dữ liệu thuộc tính, cấu hình và bằng chứng do người dùng tải từ Kaggle.
Hash xác nhận tính nhất quán file, không phải chữ ký xác thực nguồn.

## Wiki-ZSL: lần thử đầu chưa tới train

- Giữ nguyên [notebook lỗi](../historical/wikizsl_m10_failed_short_revision.ipynb) do người dùng cung cấp.
- Git báo `couldn't find remote ref b4b8420` vì bootstrap fetch một SHA rút gọn
  như remote ref. Đã sửa 3 notebook: fetch refs rồi resolve revision ở local.
- Cell cấu hình cũ thiếu M/DATA_FILE và có biến bị ghi đè; đã sửa và lần thử tiếp theo đạt.

## Wiki-ZSL: smoke test mới đã hoàn tất

- [Notebook có output](../executed/wikizsl_m10_seed300_smoke.ipynb) do người dùng cung cấp,
  SHA256 `69a835b5dd4dc3be5166ab9af58321182f6e656260d599c6bda850164d952661`.
- Papermill exception=null, duration 134.740269 giây; code thực tế b4b8420,
  Tesla T4, seed 300, m=10, smoke=true, 1 epoch.
- Snapshot 93.383 mẫu/113 quan hệ được kiểm tra, khác paper 94.383 mẫu;
  ALLOW_WIKI_COUNT_MISMATCH=True chỉ xác nhận bước kiểm thử kỹ thuật này.
- 412 train / 40 test, 103 batch; evaluation và status complete xuất thành công.
- P/R/F1 smoke: 1.2346% / 2.5% / 1.6529%. Không đưa vào bảng so sánh bài báo.
- Config hash `f4c3d4958b695f0cbd8b62fa63a2c7c74bc6612923954ef6dffd2d7ee3f26ea2`.
- [ZIP Wiki-ZSL đã nhận](wikizsl_m10_seed300_smoke_evidence.zip), 443.502 byte.
  SHA256 `5e66268924e42591a053f72211190129c12fdb8f0148f402a0a00d4c973e835a`.
  CRC và tất cả hash trong manifest hợp lệ; config khớp chính xác notebook,
  result/status/config khớp hash, history khớp và train/test rời nhau.
- Chưa kiểm tra GPU pause/resume.

## Quyết định tiếp tục

Người dùng chọn bỏ qua kiểm thử pause/resume riêng để chạy từng run chính thức.
Đã xóa notebook cấu hình thử tạm; giữ ZIP, notebook đã thực thi và lịch sử lỗi.
Cơ chế deadline/checkpoint vẫn tồn tại, nhưng GPU resume chưa được kiểm chứng.

## Run chính thức FewRel m=10 seed 300: resume đã hoàn tất

- [Notebook resume có output](../executed/fewrel_m10_seed300_resumed.ipynb),
  SHA256 `aa5ba04fc72a72eb75ac2607621b394c0dccc52a3dc376c1042436c91868b59c`.
- Code b4b84208cf43b4274f2d2aaaf14beead0daf4707, smoke=false, 10 epoch,
  config hash `bfb265a1b6c225d89a455715c727a1fb9dbc6d701f6d476e737072b083d50218`.
- Log xác nhận Resuming epoch 10, next batch 1; COMPLETE seed=300 và status complete.
- Final epoch 10: P=50.8393%, R=52.7571%, F1=51.7805%; best F1 theo log=53.4837%.
- Phiên resume mất 4489.619489 giây; Papermill exception=null.
- Đây là bằng chứng resume trên GPU ở ranh giới epoch đã chạy được. Chưa kiểm tra
  checkpoint giữa epoch m=5 hoặc so sánh toàn bộ trọng số với chạy không ngắt.
- Chờ ZIP kết quả và link version để kiểm tra đủ history/checksum và nhập results.csv.
