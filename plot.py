import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(log_csv_path):
    df = pd.read_csv(log_csv_path)
    df['episode'] = df['timesteps'] // 100

    grouped = df.groupby('episode').mean()
    # temporarily remove episodes with index >= 101 for plotting
    grouped = grouped[grouped.index < 101]

    plt.figure(figsize=(10,6))
    plt.plot(grouped.index, grouped['avg_waiting_time'], label='Avg Waiting Time')
    plt.plot(grouped.index, grouped['max_waiting_time'], label='Max Waiting Time')
    plt.plot(grouped.index, grouped['throughput'], label='Throughput')

    plt.xlabel('Episode')
    plt.ylabel('Time')
    plt.title('Training Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig("Result/20_.png")

if __name__ == "__main__":
    plot_metrics("Result/20_log.csv")
