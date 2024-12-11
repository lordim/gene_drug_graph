# Code directory for pretraining and finetuning. 

Pretraining on cold source split (GraphSAGE example): 
```python3 run_training.py configs/train/st_expanded/cold_source/gnn_cp.json outputs/pretrain --pretrain_source True ```


Finetuning on cold source split (GraphSAGE): 
``` python3 run_training.py configs/train/st_expanded/cold_source/gnn_cp.json outputs/finetune --pretrained_path_source gnn.pth ```

Pretraining on cold target split: 
```python3 run_training.py configs/train/st_expanded/cold_target/gnn_cp.json outputs/pretrain --pretrain_target True ```


Finetuning on cold target split: 
``` python3 run_training.py configs/train/st_expanded/cold_target/gnn_cp.json outputs/finetune --pretrained_path_target gnn.pth ```


