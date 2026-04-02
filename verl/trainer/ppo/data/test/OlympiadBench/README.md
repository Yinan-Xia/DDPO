---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: subfield
    dtype: string
  - name: context
    dtype: 'null'
  - name: question
    dtype: string
  - name: solution
    sequence: string
  - name: final_answer
    sequence: string
  - name: is_multiple_answer
    dtype: bool
  - name: unit
    dtype: string
  - name: answer_type
    dtype: string
  - name: error
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_bytes: 1365783
    num_examples: 675
  download_size: 635075
  dataset_size: 1365783
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
