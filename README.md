# CartPole: Q-Learning, Double Q-Learning, and DQN

Three agents learn to balance a pole on a cart: tabular Q-Learning over a discretized state space, Double Q-Learning, and a Deep Q-Network. Coursework for Reinforcement Learning at IE School of Science and Technology (Master in Computer Science and Business Technology, 2026).

## Contents

| File | Purpose |
| --- | --- |
| `Assignment3_Part1_Experiments.ipynb` | Tabular Q-Learning with sweeps over learning rate, discount factor, and state bins |
| `Assignment3_Part2_DQN_Experiments.ipynb`, `.py` | Deep Q-Network experiments |
| `Assignment3_Bonus_DoubleQLearning.ipynb` | Double Q-Learning against single Q-Learning |
| `_run_final_tuned_dqn.py`, `final_tuned.log`, `final_tuned_results.npz`, `final_tuned_dqn.weights.h5` | The final tuned DQN run, its log, results, and weights |
| `_build_*.py` | Scripts that generate the notebooks |
| `part2_run.log` | Log of the target-network sweep (the run stopped before completing) |

## Results

- DQN: solved at episode 230; greedy evaluation over 100 episodes averaged 349.10 ± 21.77
- Double Q-Learning needed 3,290 ± 1,220 episodes to solve, against 1,396 ± 404 for single Q-Learning

## Stack

Python, TensorFlow/Keras, Gymnasium, NumPy, matplotlib.
