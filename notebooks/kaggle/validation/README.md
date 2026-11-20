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
- Chưa kiểm tra Wiki-ZSL hoặc GPU pause/resume.

ZIP chứa dữ liệu thuộc tính, cấu hình và bằng chứng do người dùng tải từ Kaggle.
Hash xác nhận tính nhất quán file, không phải chữ ký xác thực nguồn.

## Wiki-ZSL: lần thử đầu chưa tới train

- Giữ nguyên [notebook lỗi](wikizsl_m10_failed_short_revision.ipynb) do người dùng cung cấp.
- Git báo `couldn't find remote ref b4b8420` vì bootstrap fetch một SHA rút gọn
  như remote ref. Đã sửa 3 notebook: fetch refs rồi resolve revision ở local.
- Cell người dùng sửa còn thiếu M/DATA_FILE, đồng thời ALLOW_WIKI_COUNT_MISMATCH
  bị gán lại False. Bản [Wiki smoke sẵn cấu hình](wikizsl_m10_smoke_ready.ipynb)
  có đủ biến, không lặp, khóa full commit b4b84208cf43b4274f2d2aaaf14beead0daf4707.
- Bản ready chỉ cho phép chênh lệch Wiki-ZSL để kiểm thử kỹ thuật trên snapshot;
  chênh lệch 93.383 so với 94.383 mẫu chưa được giải quyết cho báo cáo tái lập.
- Chưa có smoke Wiki-ZSL thành công. Cần import bản ready và chạy lại trên Kaggle.
