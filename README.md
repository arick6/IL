
## 1. Training

```bash
python train.py \
  --expert_data expert_train.pkl \
  --exp_name test0314 \
  --num_agent_train_steps_per_iter 100 \
  --n_iter 500 \
  --train_batch_size 200 \
  --max_replay_buffer_size 5000 \
  --seed 42 \
  --save_params
```

## 2. Test (pick the best checkpoint by inspecting the loss curve)
```bash
python test.py \
  --weight_path policy_itr_123.pt \
  --test_data expert_test.pkl
```
### **How to choose the best checkpoint?**

Open TensorBoard (see below), inspect the **Loss** curves logged during training, and select the iteration with the best performance. Use that checkpoint’s filename (e.g., `policy_itr_123.pt`) with `--weight_path`.

## 3. Visualisation with TensorBoard
```bash
tensorboard --logdir data
```
