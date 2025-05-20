import matplotlib.pyplot as plt 
import pickle
import os
import time
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

from infrastructure import pytorch_util as ptu
from infrastructure.logger import Logger
from infrastructure.replay_buffer import ReplayBuffer
from policies.MLP_policy import MLPPolicySL

# 创建 scaler 目录（如果不存在）
scaler_dir = "scaler"
if not os.path.exists(scaler_dir):
    os.makedirs(scaler_dir)

def run_training_loop(params):
    """
    Runs training for behavior cloning using real-world expert data
    """

    # Initialize logger
    logger = Logger(params['logdir'])

    # Set random seeds
    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    ptu.init_gpu(use_gpu=not params['no_gpu'], gpu_id=params['which_gpu'])

    # Load expert data
    print('Loading expert data from...', params['expert_data'])
    with open(params['expert_data'], 'rb') as f:
        expert_data = pickle.load(f)
    observations = np.array(expert_data["observations"])
    actions = np.array(expert_data["actions"])
    print('Done loading expert data.')

    print("observations shape: ", observations.shape)
    print("actions shape: ", actions.shape)  

    # 标准化
    obs_scaler = StandardScaler()
    act_scaler = StandardScaler()

    # 标准化数据
    observations = obs_scaler.fit_transform(observations)
    actions = act_scaler.fit_transform(actions)

    # **保存标准化器到 `scaler/` 目录**
    with open(os.path.join(scaler_dir, 'obs_scaler.pkl'), 'wb') as f:
        pickle.dump(obs_scaler, f)
    with open(os.path.join(scaler_dir, 'act_scaler.pkl'), 'wb') as f:
        pickle.dump(act_scaler, f)
    
    print(f"Scalers saved in '{scaler_dir}' directory.")

    # Initialize Replay Buffer with expert data
    replay_buffer = ReplayBuffer(params['max_replay_buffer_size'])
    replay_buffer.add_data(observations, actions)

    # Initialize policy
    ob_dim = observations.shape[-1]
    ac_dim = actions.shape[-1]
    actor = MLPPolicySL(ac_dim, ob_dim, params['n_layers'], params['size'], learning_rate=params['learning_rate'])

    # Training loop
    print("\n\n********** Starting Training ************")
    total_steps = 0
    loss_history = []
    for itr in range(params['n_iter']):
        print("\n\n********** Iteration %i ************" % itr)

        print('\nTraining agent using expert data...')
        training_logs = []
        for _ in range(params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch = replay_buffer.sample(params['train_batch_size'])
            train_log = actor.update(ob_batch, ac_batch)
            training_logs.append(train_log)

        # Log metrics
        logs = {"Iteration": itr, "Total_Steps": total_steps}
        logs.update(training_logs[-1])
        for key, value in logs.items():
            print(f'{key} : {value}')
            logger.log_scalar(value, key, itr)

        loss_history.append(training_logs[-1]['Training Loss'])

        # Save model parameters
        if params['save_params']:
            print('\nSaving model parameters...')
            actor.save(f"{params['logdir']}/policy_itr_{itr}.pt")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    min_loss = min(loss_history)
    min_loss_iter = loss_history.index(min_loss)
    plt.plot(range(len(loss_history)), loss_history, label='Training Loss', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.scatter(min_loss_iter, min_loss, color='red', label=f'Min Loss: {min_loss:.2f} (Iter {min_loss_iter})')
    plt.annotate(f'Iter {min_loss_iter}\nLoss: {min_loss:.2f}', 
                 xy=(min_loss_iter, min_loss), 
                 xytext=(min_loss_iter + len(loss_history) * 0.05, min_loss + 0.1 * min_loss),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=10, color='red')
    
    loss_plot_path = os.path.join(params['logdir'], 'loss_curve.png')
    plt.savefig(loss_plot_path)
    print(f"Loss curve saved to {loss_plot_path}")
    plt.show()
    logger.flush()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_data', '-ed', type=str, required=True,
                        help="Path to the file containing real expert data in pickle format or other supported format.")
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=100,
                        help="Number of gradient steps for training policy (per iteration in n_iter)")
    parser.add_argument('--n_iter', '-n', type=int, default=1, help="Number of training iterations")

    parser.add_argument('--train_batch_size', type=int, default=100,
                        help="Number of sampled data points to be used per gradient/train step")

    parser.add_argument('--n_layers', type=int, default=2, help="Depth of the policy to be learned")
    parser.add_argument('--size', type=int, default=64, help="Width of each layer in the policy")
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help="Learning rate for supervised learning")#[1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]


    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'bc_'
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################

    run_training_loop(params)


if __name__ == "__main__":
    main()