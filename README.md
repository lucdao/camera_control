# Headless Visual PTZ Control

A C++20, two-loop visual PTZ controller for one NVIDIA Jetson Orin Nano and one
1080p30 H.264 RTSP camera. There is no GUI.

```text
RTSP/NVDEC -> SCRFD + ArcFace -> YOLO pose + ByteTrack/body ReID -> TargetState
                                                                  |
                                   ONVIF <- visual controller 20 Hz+
```

The perception loop publishes only the latest target state. The control loop
never drains a queue of old observations. Loss of RTSP, inference, identity,
ONVIF, or a stale target causes an ONVIF Stop.

## Models

Place these fixed-shape ONNX models in `/models`:

| File | Purpose | Expected input |
| --- | --- | --- |
| `det_10g.onnx` | InsightFace buffalo_l SCRFD-10GF | `1x3x640x640` |
| `w600k_r50.onnx` | InsightFace buffalo_l ArcFace R50 | `1x3x112x112` |
| `yolo11n-pose.onnx` | person boxes + COCO-17 keypoints | `1x3x640x640` |
| `osnet_x0_25.onnx` | body ReID fallback | `1x3x256x128` |

The runtime does not download models. These four ONNX files are therefore not
present in this repository; they must be mounted at `/models`. Build/provision
them explicitly and review their licenses. TensorRT engines are
device/runtime-specific and are generated on the target Jetson:

```bash
docker compose build
docker compose run --rm ptz-control build-engines --models /models --cache /engines
```

`build-engines` now validates the actual engine tensor names, data types and
fixed shapes. It also executes SCRFD and YOLO post-processing plus ArcFace and
OSNet embedding smoke inference. A mismatched ONNX export fails at startup
instead of being accepted silently.

InsightFace's supplied pretrained packs are restricted to non-commercial
research. See [MODEL_LICENSES.md](MODEL_LICENSES.md).

## Configuration and enrollment

Copy the example and replace the RTSP/ONVIF endpoints and measured FOV table:

```bash
cp config/pipeline.example.yaml config/pipeline.yaml
export PTZ_ONVIF_USER='operator'
export PTZ_ONVIF_PASSWORD='secret'
docker compose run --rm ptz-control validate --config /config/pipeline.yaml
```

Use at least two frontal/near-frontal reference images. Enrollment rejects an
image unless exactly one face is present, its score is at least 0.70, its height
is at least 80 pixels, and its five landmarks pass the yaw check.

```bash
docker compose run --rm \
  -v "$PWD/enrollment:/data/enrollment:ro" \
  -v "$PWD/gallery:/data/gallery" \
  ptz-control enroll \
    --identity target_01 \
    --images /data/enrollment/target_01 \
    --gallery /data/gallery \
    --models /models --cache /engines
```

The normal service mounts the gallery read-only. Restart it after enrollment.

## Run

The container must use the NVIDIA runtime and host networking so it can use
Jetson decode/TensorRT and reach ONVIF/RTSP directly.

```bash
docker compose up -d
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/metrics
docker compose logs -f ptz-control
```

The camera must support `GetCapabilities`, `GetProfiles`, PTZ `GetStatus`,
`ContinuousMove`, and `Stop`. Credentials are read only from environment
variables named by the YAML; they are never placed in the YAML or logs.

## FOV calibration

For each useful zoom position, measure horizontal and vertical FOV and add a
`[zoom, hfov_degrees, vfov_degrees]` row. The controller linearly interpolates
between rows and performs a pinhole pixel-to-angle conversion. An inaccurate
table produces zoom-dependent gain error and must be corrected before tuning
PID gains.

Tune in this order:

1. Verify pan/tilt direction and emergency Stop.
2. Calibrate the FOV table with auto-zoom disabled.
3. Tune `kp`, then `kd`, then the small velocity feed-forward `kv`.
4. Verify acceleration/jerk limits and dead-zone hysteresis.
5. Enable and tune auto-zoom last.

## Development build

The dependency-free algorithm/control core can be tested on a non-Jetson host:

```bash
cmake -S . -B build \
  -DPTZ_WITH_ONVIF=OFF -DPTZ_WITH_GSTREAMER=OFF -DPTZ_WITH_TENSORRT=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The full Jetson build enables all three adapters; the supplied Dockerfile does
this and runs tests during the image build.

## Implementation and acceptance tests

The tracker is the complete tracking flow used by this project: an 8D
box/velocity Kalman filter, Hungarian global assignment, high-score association,
low-score recovery, tentative-track confirmation, lost/removed aging, and
duplicate suppression. Identity protection remains above ByteTrack: ArcFace
locks the target and OSNet is consulted when a track breaks or people cross.

On Jetson the decode path remains in NVMM: `nvv4l2decoder` produces an RGBA
NVMM surface, the surface is registered through EGL/CUDA, and CUDA kernels write
letterboxed/aligned NCHW tensors directly into TensorRT input memory. Host BGR
is retained only for replay/enrollment and non-NVMM fallback builds.

Run the automated algorithm and SOAP tests inside the build image:

```bash
ctest --test-dir build --output-on-failure
```

The SOAP test uses a local Digest-auth server and covers capability/profile
probing, status success, HTTP error, timeout, recovery, command failure,
re-probe, ContinuousMove and Stop. Controller tests separately verify the
command heartbeat, rate limiting and stale-target fail-safe.

Run every annotated replay scenario separately. The CSV format is documented
in `tests/replay/README.md`:

```bash
ptz-control replay --config /config/pipeline.yaml \
  --video /data/replay/crossing.mp4 \
  --ground-truth /data/replay/crossing.csv \
  --json /reports/crossing.json
```

Measure decoder/model throughput on the actual Orin and RTSP camera:

```bash
ptz-control benchmark --config /config/pipeline.yaml \
  --duration 60 --json /reports/orin-benchmark.json
```

The command exits non-zero unless capture is at least 29 Hz, SCRFD at least
8 Hz, pose at least 12 Hz, and capture-to-latest-inference p95 at most 200 ms.
While the real service is tracking, `/metrics` additionally reports the actual
`ptz_capture_to_command_p95_ms`, including ONVIF send completion.

The physical Stop test is opt-in because it moves the camera briefly:

```bash
ptz-control hil-stop --config /config/pipeline.yaml \
  --allow-camera-motion yes --json /reports/hil-stop.json
```

It polls ONVIF `MoveStatus` and fails if idle is not observed within 350 ms.
These numerical results cannot be truthfully generated on a development host:
the benchmark and HIL commands must be run on the target Orin Nano with the
actual ONNX files and camera. Their JSON reports are the acceptance evidence.

## Fail-safe contract

- Target older than 300 ms: Stop.
- Both face and person lost: predict for at most 250 ms, then invalidate/Stop.
- RTSP or inference failure: invalidate immediately, close/reopen with backoff.
- ONVIF failure: Stop, reprobe with backoff, never replay queued commands.
- SIGINT/SIGTERM: request both loops to stop and send a final Stop.
- Auto-zoom is disabled whenever `GetStatus` is unavailable.

`/healthz` is healthy only while frames are being delivered and ONVIF probing
has succeeded. `/metrics` exposes loop counters and both latest and rolling-p95
capture-to-command latency.
