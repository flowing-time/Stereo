# Instructions

## Environment setup

Please follow the following steps for the environment setup:

1. Create the environment
```bash
conda env create -f cs6476_proj_v2.yml
```
2. Activate the environment
```bash
conda activate cs6476_proj_v2
```

## Project code

There are two python scripts in this project zip package.

* stereo.py

This is the main python script to generate all disparity maps. Simply use the command below and all disparity maps will be generated in the "./output/" folder.
```bash
python stereo.py
```

It takes about 3 hours on my laptop to complete.

My laptop hardware configuration overview:
|                            |                      |
| -------------------------- | -------------------- |
| Model Name                 | MacBook Pro          |
| Model Identifier           | MacBookPro15,1       |
| Processor Name             | 8-Core Intel Core i9 |
| Processor Speed            | 2.3 GHz              |
| Number of Processors       | 1                    |
| Total Number of Cores      | 8                    |
| L2 Cache (per Core)        | 256 KB               |
| L3 Cache                   | 16 MB                |
| Hyper-Threading Technology | Enabled              |
| Memory                     | 16 GB                |


* FP_stereo_compare.py

This script will comare the generated disparity maps with ground truth images and give the SSIM scores.
After previous script finishes, just type the command:

```bash
python FP_stereo_compare.py
```

It will generate the table below:

| Image      | w5    | w19   | V0    | V1    | V2    |
| ---------- | ----- | ----- | ----- | ----- | ----- |
| Adirondack | 0.253 | 0.525 | 0.360 | 0.530 | 0.549 |
| Jadeplant  | 0.417 | 0.462 | 0.541 | 0.532 | 0.586 |
| Motorcycle | 0.400 | 0.609 | 0.503 | 0.566 | 0.617 |
| Piano      | 0.527 | 0.624 | 0.647 | 0.688 | 0.701 |
| Pipes      | 0.443 | 0.512 | 0.499 | 0.580 | 0.601 |
| Playroom   | 0.340 | 0.510 | 0.291 | 0.444 | 0.449 |
| Playtable  | 0.338 | 0.515 | 0.598 | 0.634 | 0.700 |
| Recycle    | 0.427 | 0.648 | 0.685 | 0.714 | 0.733 |
| Shelves    | 0.315 | 0.521 | 0.584 | 0.612 | 0.651 |
| Teddy      | 0.372 | 0.513 | 0.492 | 0.526 | 0.526 |
| Vintage    | 0.637 | 0.797 | 0.673 | 0.738 | 0.797 |

