# Code directory for pretraining and finetuning. 

Pretraining: 
```python3 run_training.py configs/train/st_expanded/cold_source/gnn_cp.json outputs/pretrain --pretrain_source True ```


Finetuning: 
``` python3 run_training.py configs/train/st_expanded/cold_source/gnn_cp.json outputs/pretrain --pretrained_path gnn.pth ```



