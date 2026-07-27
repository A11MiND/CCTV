# Third-Party Notices

Author: Edcosys

## Supermarket demo footage

- Title: Customers Shopping at Supermarket
- Creator: Suika Chan
- Source: https://www.pexels.com/video/customers-shopping-at-supermarket-10901926/
- License: https://www.pexels.com/license/
- Local source file: `assets/video/pexels-hong-kong-supermarket.mp4`
- Derived file: `public/assets/video/yolo26-retail-demo.mp4`

The derived file adds YOLO26s person detections, ByteTrack track IDs, runtime information, and source attribution.

## Ultralytics YOLO

The reproducible demo script uses the separately installed `ultralytics` Python package and a locally downloaded YOLO26 model. Package and model files are excluded from this repository.

- Project: https://github.com/ultralytics/ultralytics
- Model documentation: https://docs.ultralytics.com/models/yolo26/
- Licensing: https://www.ultralytics.com/license

Review the Ultralytics AGPL-3.0 and Enterprise licensing options before commercial deployment or distribution.

## Web prototype dependencies

The prototype uses React, React DOM, Vite, and the Vite React plugin. Their exact resolved versions are recorded in `pnpm-lock.yaml`. License texts remain with the respective packages installed through the package manager.

## Shoplifting benchmark dataset

The benchmark demo and reported metrics use:

- Title: Shoplifting Dataset (2022) - CV Laboratory MNNIT Allahabad
- Creators: Mohd. Aquib Ansari and Dushyant Kumar Singh
- Source: https://data.mendeley.com/datasets/r3yjf35hzr/1
- DOI: https://doi.org/10.17632/r3yjf35hzr.1
- License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Kaggle re-upload: https://www.kaggle.com/datasets/kipshidze/shoplifting-video-dataset

Dataset files and pretrained model weights are excluded from this repository.
The derived benchmark video adds model probabilities, YOLO26s person boxes,
ByteTrack IDs, evaluation context, and attribution.

## Other referenced public datasets

Public datasets discussed in the proposal are referenced for evaluation and development planning. Each download and derivative model must follow the source dataset terms.

- RetailAction: https://huggingface.co/datasets/standard-cognition/RetailAction
- PoseLift: https://github.com/TeCSAR-UNCC/PoseLift
- UCF-Crime: https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/
- DCSASS: https://www.kaggle.com/datasets/mateohervas/dcsass-dataset
- CHAD: https://arxiv.org/abs/2212.09258
