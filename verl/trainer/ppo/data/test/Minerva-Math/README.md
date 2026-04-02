---
dataset_info:
  features:
  - name: problem
    dtype: string
  - name: solution
    dtype: string
  - name: type
    dtype: string
  - name: idx
    dtype: int64
  splits:
  - name: train
    num_bytes: 227367
    num_examples: 272
  download_size: 102139
  dataset_size: 227367
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
