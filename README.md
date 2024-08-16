# An EEG Classification Method for Alzheimer’s Disease Based on Self-Supervised Contrastive Learning and Time-Frequency Domain Analysis
## Introduction

This project utilizes a transfer learning approach to classify EEG data. Initially, self-supervised contrastive learning is employed for pre-training on an epilepsy EEG dataset. Following this, the model is fine-tuned using labeled Alzheimer's Disease (AD) and Cognitively Normal (CN) data to accomplish the EEG classification task.
This library is constructed based on the following repository:  
[Time-Series-Library](https://github.com/thuml/Time-Series-Library)
## Usage

Follow these steps to process the data, perform pre-training, and fine-tune the model:

1. **Data Preprocessing:**
   - Use the `forTUH_EDF.py` script located in the `data_preprocessing` folder to preprocess the EDF data downloaded from the TUH corpus.


2. **Self-Supervised Contrastive Learning Pre-training:**
   - Run the `pretrain.py` script to perform self-supervised contrastive learning on the preprocessed epilepsy EEG dataset.


3. **Supervised Learning and Evaluation:**
   - Finally, use the `Finetune.py` script for supervised learning and evaluation using the labeled AD and CN data.
   - You can use the pre-trained models from the `pre-trained model` folder by specifying the path in `Finetune.py`.


### Pre-trained Models

- The `pre-trained model` folder contains models that have already been pre-trained. You can directly use these models in the `Finetune.py` script for fine-tuning and evaluation.

## Dataset

The dataset used in this project can be obtained from the following source:

- **TUH EEG Corpus:** [Link to the dataset](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)
- **AD EEG Dataste:**[Link to the dataset](https://openneuro.org/datasets/ds004504/versions/1.0.7)

Please refer to the dataset provider's website for instructions on how to download and access the data.
-The AD dataset used in this project has already been preprocessed. The preprocessing involved segmenting the original data into sub-samples, each of which is saved as a CSV file.

### Preprocessing Details:
- Each sub-sample corresponds to a segment of the original EEG data.
- The files are named in the format `sub001_segment1.csv`, where `sub001` indicates the first subject and `segment1` indicates the first sub-sample for that subject.

For example:
- `sub001_segment1.csv` represents the first sub-sample of the first subject.
- `sub002_segment3.csv` represents the third sub-sample of the second subject.

This preprocessed dataset can be used directly in the subsequent stages of the project, such as model training and evaluation.

## Environment

This project was developed and tested in the following environment:

- **Python version**: 3.8.19
- **Key dependencies**:
  - torch==1.10.1+cu111
  - torchaudio==0.10.1+cu111
  - torchvision==0.11.2+cu111
  - numpy==1.23.5
  - scikit-learn==1.2.2
  - pandas==1.5.3
  - matplotlib==3.7.0
  - mne==1.6.1
  - requests==2.32.3

For a full list of dependencies, please refer to the `requirements.txt` file.

## Installation

To set up the environment and install all required dependencies, you can use the following command:

```bash
pip install -r requirements.txt
