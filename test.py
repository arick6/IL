import pickle
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from policies.MLP_policy import MLPPolicySL  # 根据你的路径导入 MLPPolicySL

def test_policy(params):
    """
    Test the trained policy using provided test data and visualize results.
    """
    # Load trained policy
    model_path = params['weight_path']
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist. Make sure training is complete.")
        return

    print(f"Loading policy from {model_path}")

    # Load test data
    print('Loading test data from...', params['test_data'])
    with open(params['test_data'], 'rb') as f:
        test_data = pickle.load(f)

    # Dynamically determine observation and action dimensions
    test_observations = np.array(test_data["observations"])
    test_actions = np.array(test_data["actions"])
    
    ob_dim = test_observations.shape[1]  # Number of features in observations
    ac_dim = test_actions.shape[1]  # Number of features in actions

    print(f"Detected observation dimension: {ob_dim}, action dimension: {ac_dim}")

    # Initialize the actor with dynamic dimensions
    actor = MLPPolicySL(ac_dim, ob_dim, params['n_layers'], params['size'])
    actor.load_state_dict(torch.load(model_path))
    actor.eval()  # Set the model to evaluation mode

    print("Test observations shape:", test_observations.shape)
    print("Test actions shape:", test_actions.shape)

    # Load scalers (自动寻找 scaler 目录)
    scaler_dir = next((d for d in os.listdir('.') if os.path.isdir(d) and "scaler" in d), "scaler")
    with open(os.path.join(scaler_dir, 'obs_scaler.pkl'), 'rb') as f:
        obs_scaler = pickle.load(f)
    with open(os.path.join(scaler_dir, 'act_scaler.pkl'), 'rb') as f:
        act_scaler = pickle.load(f)

    # Standardize test observations
    test_observations_scaled = obs_scaler.transform(test_observations)

    # Start testing
    total_test_error = 0
    num_test_samples = len(test_observations)
    errors = []
    mse_losses = []  # 用于存储 MSE 损失
    predicted_actions_all = []
    true_actions_all = []

    for i in range(num_test_samples):
        obs = test_observations_scaled[i]
        true_action = test_actions[i]

        # Get action from the policy
        predicted_action_scaled = actor.get_action(obs)

        # Ensure predicted action is correctly shaped before inverse transform
        if predicted_action_scaled.ndim == 1:
            predicted_action_scaled = predicted_action_scaled.reshape(1, -1)

        # Inverse transform to original scale
        predicted_action = act_scaler.inverse_transform(predicted_action_scaled)[0]

        # Compute L2 norm (Euclidean distance) as error
        error = np.linalg.norm(predicted_action - true_action)
        total_test_error += error
        errors.append(error)

        # Compute MSE loss for this sample
        mse_loss = np.mean((predicted_action - true_action) ** 2)
        print(mse_loss)
        mse_losses.append(mse_loss)  # 记录损失值

        # Store for visualization
        predicted_actions_all.append(predicted_action)
        true_actions_all.append(true_action)

    # Compute average error
    avg_error = total_test_error / num_test_samples
    print(f"\nAverage Test Error: {avg_error:.4f}")

    # Convert lists to numpy arrays
    predicted_actions_all = np.array(predicted_actions_all)
    true_actions_all = np.array(true_actions_all)
    
    # Plot L2 error over time
    plt.figure(figsize=(10, 5))
    plt.plot(errors, label="L2 Error", color="red")
    plt.xlabel("Test Sample Index")
    plt.ylabel("L2 Error")
    plt.title("Prediction Error Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("error_curve.png")  # Save figure for paper use
    plt.show()

    # Plot MSE loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(mse_losses, label="MSE Loss", color="blue")
    plt.xlabel("Test Sample Index")
    plt.ylabel("MSE Loss")
    plt.title("MSE Loss Curve Over Test Samples")
    plt.legend()
    plt.grid(True)
    plt.savefig("mse_loss_curve.png")  # Save figure for paper use
    plt.show()

    # Plot action comparison for each dimension
    action_labels = ["position", "force"]

    plt.figure(figsize=(10, 5))
    for dim in range(ac_dim):
        plt.plot(true_actions_all[:, dim], label=f"True {action_labels[dim]}", linestyle="dashed")
        plt.plot(predicted_actions_all[:, dim], label=f"Predicted {action_labels[dim]}", alpha=0.7)
    
    plt.xlabel("Test Sample Index")
    plt.ylabel("Action Value")
    plt.title("True vs Predicted Actions")
    plt.legend()
    plt.grid(True)
    plt.savefig("action_comparison.png")  # Save figure for paper use
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', '-wp', type=str, required=True,
                        help="Path to the pre-trained model weights file.")
    parser.add_argument('--test_data', '-td', type=str, required=True,
                        help="Path to the file containing test data in pickle format.")

    # Other arguments...
    parser.add_argument('--n_layers', type=int, default=2, help="Depth of the policy to be learned")
    parser.add_argument('--size', type=int, default=64, help="Width of each layer in the policy")
    args = parser.parse_args()

    # Convert args to dictionary
    params = vars(args)

    ##################
    ### LOAD & TEST
    ##################
    test_policy(params)

if __name__ == "__main__":
    main()
