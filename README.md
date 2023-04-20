# Weakly-Supervised Anomaly Detection for CT Scans

## About The Project
This project presents a novel weakly supervised anomaly detection (WSAD) algorithm for CT scans, which reduces the annotation workload while providing better performance than conventional methods. 

The proposed WSAD algorithm is trained based on scan-wise normal and anomalous annotations, unlike slice-level annotations required in conventional supervised learning. The methodology is motivated by video anomaly detection tasks.

## Getting Started


### Requirements
- Python 3.9
- PyTorch 1.13.0
- Cuda 11.7
- Pytorch Image Models (timm)
- Pydicom
- OpenCV
- pandas
- scikit-learn

### Dataset 

Download the following dataasets. Both of them are publicly availablle.

- [RSNA brain hemorrhage dataset](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data): Brain CT dataset collected from patients with intracranial hemorrhages. The data named "stage_2_train" are only required as they contain both training and testing samples.
- [COVID-CTset](https://github.com/mr7495/COVID-CTset): Lung CT dataset collected from patients with COVID-19.

## Pre-processing
### Brain CT dataset
Place downloaded data in a local directory (e.g., /path_to/rsna-intracranial-hemorrhage-detection) then run the following six scripts one by one for pre-processing downloaded data.

- [01_dcm2png.py](./prepare_dataset_brain/01_dcm2png.py) conversts dcm to png files.
- [02_renamepng.py](./prepare_dataset_brain/02_renamepng.py) organizes png files to several directories.
- [03_makelabelfile.py](./prepare_dataset_brain/03_makelabelfile.py) creates csv file specifiying groundtruth labels.
- [04_brainextract.py](./prepare_dataset_brain/04_brainextract.py) perfomrs skull stipping and segments brain regions.
- [05_makemaskimg.py](./prepare_dataset_brain/05_makemaskimg.py): creates mask images based on extracted brain regions.
- [06_feature_extractor.py](./prepare_dataset_brain/06_feature_extractor.py) extract features from pretrained models.

### Lung CT dataset
Place downloaded data in a local directory (e.g., /path_to/COVID-CTset) then run the following five scripts one by one for pre-processing downloaded data.

- [01_renamepng.py](./prepare_dataset_lung/01_renamepng.py) renames downloaded files in an organized manner.
- [02_makelabelfile.py](./prepare_dataset_lung/02_makelabelfile.py) creates csv file specifiying groundtruth labels.
- [03_lungextract.py](./prepare_dataset_lung/03_lungextract.py) segments lung regions.
- [04_makemaskimg.py](./prepare_dataset_lung/04_makemaskimg.py) creates mask images based on extracted lung regions.
- [05_feature_extractor.py](./prepare_dataset_lung/05_feature_extractor.py) extract features from pretrained models.

## Training a model
1. Set parameters at [parameters.py](./parameters.py)

2. Run [run_train.py](./run_train.py)

## Testing a model
1. Run [run_test.py](./run_test.py)

2. A performance report will be generated.