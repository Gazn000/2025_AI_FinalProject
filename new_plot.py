import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_avg_waiting_times(log_files, path):
    plt.figure(figsize=(10,6))

    for algo_name, file_path in log_files.items():
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
        
        df = pd.read_csv(file_path)

        if 'episode' not in df.columns and 'timesteps' in df.columns:
            df['episode'] = df['timesteps'] // 100

        grouped = df.groupby('episode')['avg_waiting_time'].mean()
        grouped = grouped[grouped.index < 101]

        plt.plot(grouped.index, grouped.values, label=algo_name)

    plt.xlabel("Episode")
    plt.ylabel("Average Waiting Time (s)")
    plt.title("Average Waiting Time Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)

def plot_throughput(log_files, path):
    plt.figure(figsize=(10,6))

    for algo_name, file_path in log_files.items():
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
        
        df = pd.read_csv(file_path)

        if 'episode' not in df.columns and 'timesteps' in df.columns:
            df['episode'] = df['timesteps'] // 100

        grouped = df.groupby('episode')['throughput'].mean()
        grouped = grouped[grouped.index < 101]

        plt.plot(grouped.index, grouped.values, label=algo_name)

    plt.xlabel("Episode")
    plt.ylabel("Vehicles")
    plt.title("Throughput Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)

def plot_reward(log_files, path):
    plt.figure(figsize=(10,6))

    for algo_name, file_path in log_files.items():
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
        
        df = pd.read_csv(file_path)

        if 'episode' not in df.columns and 'timesteps' in df.columns:
            df['episode'] = df['timesteps'] // 100

        grouped = df.groupby('episode')['reward'].mean()
        grouped = grouped[grouped.index < 101]

        plt.plot(grouped.index, grouped.values, label=algo_name)

    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.title("Reward Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
def plot_max_waiting_time(log_files, path):
    plt.figure(figsize=(10,6))

    for algo_name, file_path in log_files.items():
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
        
        df = pd.read_csv(file_path)

        if 'episode' not in df.columns and 'timesteps' in df.columns:
            df['episode'] = df['timesteps'] // 100

        grouped = df.groupby('episode')['max_waiting_time'].mean()
        grouped = grouped[grouped.index < 101]

        plt.plot(grouped.index, grouped.values, label=algo_name)

    plt.xlabel("Episode")
    plt.ylabel("Max Waiting Time (s)")
    plt.title("Max Waiting Time Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
if __name__ == "__main__":
    log_files = {
        "DQN": "2x2_result/DQN_0001.csv",
        "QR-DQN": "2x2_result/QR_DQN_log_0001.csv",
        "PPO": "2x2_result/ppo_log_0001.csv",
        "A2C": "2x2_result/a2c_log_0001.csv",
    }
    save_path = {
        "avg_waiting_time": "Plot_result/2x2_avg_0001.png",
        "max_waiting_time": "Plot_result/2x2_max_0001.png",
        "throughput": "Plot_result/2x2_throughput_0001.png",
        "reward": "Plot_result/2x2_reward_0001.png"
    }
    plot_avg_waiting_times(log_files, save_path["avg_waiting_time"])
    plot_max_waiting_time(log_files, save_path["max_waiting_time"])
    plot_throughput(log_files, save_path["throughput"])
    plot_reward(log_files, save_path["reward"])
