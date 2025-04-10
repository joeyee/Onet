# Onet Project

## Overview
This project implements Onet (unsupervised semantic segmentation) for various scenarios including:
- Simulated Rayleigh distributed clutter
- NAU-Rain case
- ZY3 remote sensing datasets

## Features
- Fast training capability (11 epochs for ZY3 datasets)
- High accuracy (OA: 0.9254, mIoU: 0.7958)
- Support for multiple datasets
- Pre-processing techniques for enhanced performance

## Installation
```bash
git clone https://github.com/yourusername/onet_github.git
cd onet_github
```

## Usage
Please refer to the [using_manual.md](using_manual.md) for detailed usage instructions.

## Project Structure
```
onet_github/
├── configs/                 # Configuration files
├── dataloader/             # Data loading utilities
├── Onet_vanilla_20240606.py    # Main ONET model implementation
├── Train_Onet_on_simclutter_20250407.py    # Training for simulated clutter
├── Train_Onet_on_zy3_20240606.py    # Training for ZY3 datasets
├── exp_nau_rain_20240513.py    # NAU-Rain experiment
└── utils_20231218.py      # Utility functions
```

## License
[Your chosen license]

## Contact
[Your contact information] 