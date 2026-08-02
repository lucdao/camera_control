# Replay ground truth

Replay clips are deliberately not committed. A CSV uses original video pixels:

```csv
frame,visible,cx,cy
1,1,960,300
2,1,968,301
3,0,0,0
```

`visible=1` means the configured target identity is visible. `cx,cy` is the
expected visual aim point (face landmark or torso anchor), not the raw bounding
box centre. Run:

```bash
ptz-control replay --config /config/pipeline.yaml \
  --video /data/replay/crossing.mp4 \
  --ground-truth /data/replay/crossing.csv \
  --json /reports/crossing.json
```

The command fails unless target recall is at least 90%, false-positive rate is
at most 2%, and there are zero target track-ID switches.
