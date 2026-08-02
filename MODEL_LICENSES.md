# Model licensing notice

This repository does not redistribute model weights.

- InsightFace states that the pretrained model packs it supplies, including
  `buffalo_l`, are available for non-commercial research purposes. Obtain a
  separately licensed replacement before commercial deployment.
- YOLO, OSNet, exported ONNX weights, training datasets, and export tooling may
  carry independent code, model, or dataset terms. Review the exact artifacts
  placed in `/models`; this project does not grant rights to them.

The C++ model adapters intentionally use file-based interfaces so licensed ONNX
replacements can be substituted without changing the control pipeline.
