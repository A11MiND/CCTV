# YOLO26 Boxed Training Videos

Author: Edcosys

These three H.264 videos run YOLO26s person detection and ByteTrack tracking
on source videos from the actor/clothing-disjoint training split. They display
person boxes, confidence values, track IDs, trajectories, measured GPU latency,
and dataset attribution.

| Video | Class | Frames | Pipeline FPS | Inference p50 | Inference p95 | Tracks |
|---|---|---:|---:|---:|---:|---:|
| `shoplifting-8-yolo26.mp4` | Shoplifting | 158 | 38.12 | 12.28 ms | 13.78 ms | 1 |
| `normal-6-yolo26.mp4` | Normal | 173 | 39.58 | 12.26 ms | 14.31 ms | 4 |
| `shoplifting-1-yolo26.mp4` | Shoplifting | 156 | 40.26 | 11.92 ms | 13.75 ms | 4 |

Runtime: NVIDIA GeForce RTX 4060 Laptop GPU, YOLO26s, ByteTrack, CUDA,
640-pixel inference input, 15 FPS output.

Dataset:

- Source: https://data.mendeley.com/datasets/r3yjf35hzr/1
- DOI: https://doi.org/10.17632/r3yjf35hzr.1
- License: CC BY 4.0

The boxes are perception output. The source class comes from the dataset label;
YOLO26 does not independently determine theft or intent in these videos.
