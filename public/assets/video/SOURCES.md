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

The derived clip adds R3D-18 temporal-classification probabilities, the fixed
0.50 decision threshold, YOLO26s person detections, ByteTrack track IDs,
aggregate held-out metrics, and source attribution. It is a simulated
laboratory scenario from the public dataset.
