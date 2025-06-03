# 2025_AI_FinalProject
This is the final project for the 2025 Spring Intro to AI course. It uses [CityFlow](https://cityflow-project.github.io) to simulate city traffic and trains AI models to optimize traffic light control.
## System Requirements
- Python 3.6
- pip/conda
- Linux/macOS/Windows (recommend to use Linux and WSL2)
## Installation
CityFlow environment setup please refer to [Installation](https://cityflow.readthedocs.io/en/latest/install.html)
1. Clone the project:
```bash
git clone https://github.com/Gazn000/2025_AI_FinalProject.git
cd 2025_AI_FinalProject
```
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```
  If you are using Docker, most dependencies are already included in the image.
You only need to install a few additional Python packages inside the container:
  ```bash 
  pip install gym stable_baselines3
  ```
3. Verify CityFlow installation:
```bash
python -c "import cityflow"
````
## Project Structure
```plaintxt
2025_AI_FinalProject/
├── Plot_result/                         # Folder for storing result plots (e.g., reward curves)
├── gym_cityflow/                        # Custom Gym-compatible environment for CityFlow
│   ├── __init__.py
│   └── envs/
│       ├── 1x1_config/                  # Configuration files for 1×1 road network
│       ├── 1x3_config/                  # Configuration files for 1×3 road network
│       ├── 2x2_config/                  # Configuration files for 2×2 road network
│       ├── CityFlow_1x1_LowTraffic.py   # Environment script for 1×1 with low traffic
│       ├── CityFlow_1x3_LowTraffic.py
│       ├── CityFlow_2x2_LowTraffic.py
│       ├── __init__.py
│       └── README.md
├── .gitignore                           # Git ignore rules
├── README.md                            # Main documentation
├── requirements.txt                     # Python dependencies list
├── setup.py                             # Optional setup script
├── test.py                              # Basic test or baseline run
├── test_A2C.py                          # Training and testing with A2C algorithm
├── test_DQN.py                          # Training and testing with DQN
├── test_PPO.py                          # Training and testing with PPO
├── test_QR_DQN.py                       # Training and testing with QR-DQN
├── compare_1x1.py                       # Compare models on 1×1 network
├── compare_1x3.py
├── compare_2x2.py
├── compare_reward.py                    # Compare reward trends across models
├── compare_self.py                      # Custom comparison script
├── new_plot.py                          # Additional plotting script
├── new_plot_reward.py                   # Additional reward plotting script

```
## Usage
### Data
You can find `1x1_config`, `2x2_config`, `1x3_config` folders.  
According to you roadnet size to change the specific folder of roadnet.json and flow.json. (Guangfu Rd. is 1x1 roadnet)  
Check your config.json dir in the folder:
```json
{
    "interval": 1.0,
    "seed": 1,
    "dir": "/gym_cityflow/envs/1x1_config/",
    "roadnetFile": "roadnet.json",
    "flowFile": "flow.json",
    "rlTrafficLight": true,
    "saveReplay": true,
    "roadnetLogFile": "roadnetlog.json",
    "replayLogFile": "replay.txt"
}

```
### Result
Each `test_XXX.py` is a different model, to visualize training results of each model:
1. Change the dir to the correct roadnet folder name: line 49 
```python
self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1x1_config")
```
2. Visualize with frontend
Find `frontend` html in [CityFlow_Github](https://github.com/cityflow-project/CityFlow)  
Upload roadnetlog.json and replay.txt and play start.

![image](https://github.com/user-attachments/assets/c1222943-994e-4fd3-b53f-ed675ba01f0e)

## Hyperparameters
- policy = "MlpPolicy"
- Learning rate = 0.0000625
- log_interval = 10
- total_episode = 100
- Other hyperparameters are set as default

## Experiment Results
[All Plot Results](Plot_result/)
### Results on Intersection of Guangfu and Daxue Road

The following figure compares 4 key metrics across different algorithms.  
![image](https://github.com/user-attachments/assets/c98f2686-a756-42c1-aefb-414a3a649918)

The **X-axis** shows the number of training episodes.  
The **Y-axis** shows the corresponding metric for each plot.

#### Average Waiting Time
- PPO and A2C maintain low waiting times throughout training.
- DQN shows improvement during training, but remains less stable.

#### Max Waiting Time
- PPO and A2C remain relatively stable but do not reach as low as DQN.
- QR-DQN shows large oscillations and performs the worst on this metric.
#### Throughput Comparison
- Except for DQN, other models show similar throughput.
- Throughput exhibits periodic changes due to bursty traffic patterns.
#### Reward Comparison
- PPO, A2C and DQN quickly reach stable and high rewards.
- QR-DQN show large oscillations, indicating unstable learning.

#### Summary
PPO and A2C outperform DQN and QR-DQN overall.  
However, DQN shows clear improvement during training.

#### Interesting Finding on Guangfu & Daxue Intersection
DQN shows a clear downward trend in both average waiting time and max waiting time. By the end of training, DQN outperforms PPO and A2C on these two metrics! This suggests that DQN may have discovered a more effective policy for this particular traffic pattern.

## Reference
- [CityFlow](https://cityflow-project.github.io)
- [gym](https://www.gymlibrary.dev/index.html)
- [Deep Reinforcement Learning for Traffic Signal Control](https://ieeexplore.ieee.org/document/9241006?denied=)
- [Reinforcement Learning for Traffic Signal Control](https://traffic-signal-control.github.io)
