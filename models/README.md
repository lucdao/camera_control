# Model provenance

- `det_10g.onnx` and `w600k_r50.onnx` were extracted from InsightFace's
  official `buffalo_l.zip` v0.7 release:
  `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip`.
  The archive SHA-256 was
  `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f`.
  Their ONNX input metadata was fixed to `1x3x640x640` and `1x3x112x112`
  respectively; graph weights were not changed.
- `yolo11n-pose.onnx` was exported at static `1x3x640x640`, opset 17, from
  Ultralytics' official `yolo11n-pose.pt` asset using Ultralytics 8.4.115:
  `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt`.
  Source checkpoint SHA-256:
  `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`.
- `osnet_x0_25.onnx` was exported at static `1x3x256x128`, opset 17, from
  the official Torchreid MSMT17-combineall checkpoint:
  `https://drive.google.com/file/d/1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e/view`.
  Torchreid source commit:
  `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`; checkpoint SHA-256:
  `cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18`.

All four files pass `onnx.checker.check_model` and CPU smoke inference. See the
repository `MODEL_LICENSES.md` before redistribution or commercial use.
