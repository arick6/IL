Train: python train.py --expert_data expert_train.pkl --exp_name test0314 --num_agent_train_steps_per_iter 100 --n_iter 500 --train_batch_size 200 --max_replay_buffer_size 5000 --seed 42 --save_params
Test(need to choose the best weight by loss curve): python test.py --weight_path policy_itr_123.pt --test_data expert_test.pkl
visualize your runs using tensorboard: tensorboard --logdir data
