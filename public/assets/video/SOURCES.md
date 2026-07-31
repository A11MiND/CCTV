# Demo Video Sources

## Hong Kong Supermarket Footage

- Source page: https://www.pexels.com/video/customers-shopping-at-supermarket-10901926/
- Creator: Suika Chan
- License: https://www.pexels.com/license/
- Local source: `assets/video/pexels-hong-kong-supermarket.mp4`
- Derived demo: `public/assets/video/yolo26-retail-demo.mp4`

The derived clip demonstrates YOLO26s person detection and ByteTrack tracking. The people in the source footage are ordinary supermarket customers. The interface event sequence is a product-workflow simulation.

## Shoplifting Benchmark Footage

- Dataset: Shoplifting Dataset (2022) - CV Laboratory MNNIT Allahabad
- Creators: Mohd. Aquib Ansari and Dushyant Kumar Singh
- Source: https://data.mendeley.com/datasets/r3yjf35hzr/1
- DOI: https://doi.org/10.17632/r3yjf35hzr.1
- License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Kaggle re-upload: https://www.kaggle.com/datasets/kipshidze/shoplifting-video-dataset
- Derived demo: `public/assets/video/shoplifting-mil-heldout-demo.mp4`
- YOLO26 v2 derived demo: `public/assets/video/shoplifting-yolo26-v2-demo.mp4`
- YOLO26 boxed training videos:
  - `public/assets/video/training-boxed/shoplifting-8-yolo26.mp4`
  - `public/assets/video/training-boxed/normal-6-yolo26.mp4`
  - `public/assets/video/training-boxed/shoplifting-1-yolo26.mp4`

The derived clip adds R3D-18 temporal-classification probabilities, the fixed
0.50 decision threshold, YOLO26s person detections, ByteTrack track IDs,
aggregate held-out metrics, and source attribution. It is a simulated
laboratory scenario from the public dataset.

The v2 clip adds YOLO26 person/bag perception, a full-frame plus person-tube
MViT score, the R3D-18 safety component, and the recall-first maximum risk.
Its 93.3% result is an opened-test development benchmark.

The boxed training videos use source clips from the training split. They add
YOLO26s person boxes, ByteTrack IDs, trajectories, measured GPU latency, and
dataset attribution. Detailed runtime results and previews are in
`docs/results/training-boxed/`.
