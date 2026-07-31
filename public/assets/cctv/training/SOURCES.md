# Training Dataset Camera Previews

Author: Edcosys

These UI camera previews are six-second, muted H.264 derivatives of videos in
the actor/clothing-disjoint **training split**. They replace generated CCTV
stills in the live-operations prototype.

Dataset:

- Title: Shoplifting Dataset (2022) - CV Laboratory MNNIT Allahabad
- Creators: Mohd. Aquib Ansari and Dushyant Kumar Singh
- Source: https://data.mendeley.com/datasets/r3yjf35hzr/1
- DOI: https://doi.org/10.17632/r3yjf35hzr.1
- License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

| UI set | Derived file stem | Source video | Split | Actor group | Class | Segment |
|---|---|---|---|---|---|---|
| A Mixed | `set-a-shoplifting-8` | `shoplifting/shoplifting-8.mp4` | train | A_checkered | Shoplifting | 00:02-00:08 |
| A Mixed | `set-a-normal-6` | `normal/normal-6.mp4` | train | C_purple | Normal | 00:02-00:08 |
| A Mixed | `set-a-shoplifting-1` | `shoplifting/shoplifting-1.mp4` | train | D_navy_stripe | Shoplifting | 00:02-00:08 |
| B Positive | `set-b-shoplifting-10` | `shoplifting/shoplifting-10.mp4` | train | B_light_blue | Shoplifting | 00:02-00:08 |
| B Positive | `set-b-shoplifting-14` | `shoplifting/shoplifting-14.mp4` | train | C_purple | Shoplifting | 00:02-00:08 |
| B Positive | `set-b-shoplifting-16` | `shoplifting/shoplifting-16.mp4` | train | D_navy_stripe | Shoplifting | 00:02-00:08 |
| C Normal | `set-c-normal-2` | `normal/normal-2.mp4` | train | B_light_blue | Normal | 00:02-00:08 |
| C Normal | `set-c-normal-7` | `normal/normal-7.mp4` | train | D_navy_stripe | Normal | 00:02-00:08 |
| C Normal | `set-c-normal-11` | `normal/normal-11.mp4` | train | E_older_plaid | Normal | 00:02-00:08 |

Each `.jpg` poster is taken at 00:05 of the same source video. The footage is
a staged laboratory dataset and is presented as development material, not as
a live store incident or production evidence.
