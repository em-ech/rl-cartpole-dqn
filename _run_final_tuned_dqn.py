"""Final tuned DQN run (24/24, no target net, seed=42) + 100-episode greedy eval.

Standalone version of the corresponding cell in
`Assignment3_Part2_DQN_Experiments.ipynb`. Run after the width sweep
established 24/24 as the clear winner.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import random
from collections import deque
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

STATE_SIZE = 4
ACTION_SIZE = 2
SOLVED_THRESHOLD = 195.0
SOLVED_WINDOW = 100


class DQNAgent:
    def __init__(self, hidden_width=24, gamma=0.99, lr=0.001,
                 eps_start=1.0, eps_min=0.01, eps_decay=0.995,
                 batch_size=64, memory_size=2000,
                 use_target=False, target_update_C=200, seed=0):
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.use_target = use_target
        self.target_update_C = target_update_C
        self.train_steps = 0
        self.rng = np.random.default_rng(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.model = self._build_model(hidden_width, lr)

    def _build_model(self, hidden_width, lr):
        inp = Input(shape=(STATE_SIZE,))
        x = Dense(hidden_width, activation='relu')(inp)
        x = Dense(hidden_width, activation='relu')(x)
        out = Dense(ACTION_SIZE, activation='linear')(x)
        m = Model(inp, out)
        m.compile(loss='mse', optimizer=Adam(learning_rate=lr))
        return m

    def act(self, state):
        if self.rng.random() <= self.eps:
            return int(self.rng.integers(0, ACTION_SIZE))
        q = self.model.predict(state, verbose=0)
        return int(np.argmax(q[0]))

    def remember(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, float(terminated)))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        indices = self.rng.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        states_arr      = np.vstack(states)
        next_states_arr = np.vstack(next_states)
        rewards_arr     = np.asarray(rewards, dtype=np.float32)
        dones_arr       = np.asarray(dones, dtype=np.float32)
        actions_arr     = np.asarray(actions, dtype=np.int64)
        target_qs = self.model.predict(states_arr, verbose=0)
        next_qs = self.model.predict(next_states_arr, verbose=0)
        target_qs[np.arange(self.batch_size), actions_arr] = (
            rewards_arr + self.gamma * np.max(next_qs, axis=1) * (1.0 - dones_arr)
        )
        self.model.fit(states_arr, target_qs, epochs=1, verbose=0)
        self.train_steps += 1

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


def train_dqn(agent, max_episodes=400, max_steps=500, replay_every=4):
    env = gym.make('CartPole-v1')
    rewards = []
    solved_at = None
    step_counter = 0
    t0 = time.perf_counter()
    for ep in range(max_episodes):
        state, _ = env.reset(seed=int(agent.rng.integers(0, 2**31)))
        state = np.reshape(state, [1, STATE_SIZE])
        total = 0.0
        for _ in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, STATE_SIZE])
            agent.remember(state, action, reward, next_state, terminated)
            state = next_state
            total += reward
            step_counter += 1
            if len(agent.memory) >= agent.batch_size and step_counter % replay_every == 0:
                agent.replay()
            if done:
                break
        agent.decay_eps()
        rewards.append(total)
        if (ep + 1) % 20 == 0:
            roll = np.mean(rewards[-min(SOLVED_WINDOW, len(rewards)):])
            print(f"  ep {ep + 1:3d} | last20={np.mean(rewards[-20:]):6.1f} | "
                  f"roll100={roll:6.1f} | eps={agent.eps:.3f}", flush=True)
        if solved_at is None and len(rewards) >= SOLVED_WINDOW:
            if np.mean(rewards[-SOLVED_WINDOW:]) >= SOLVED_THRESHOLD:
                solved_at = ep + 1
                print(f"  SOLVED at ep {solved_at}", flush=True)
                break
    elapsed = time.perf_counter() - t0
    env.close()
    final_avg = (float(np.mean(rewards[-SOLVED_WINDOW:]))
                 if len(rewards) >= SOLVED_WINDOW else float(np.mean(rewards)))
    return rewards, solved_at, final_avg, elapsed


def evaluate(agent, n_episodes=100, seed=12345):
    env = gym.make('CartPole-v1')
    rng = np.random.default_rng(seed)
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        state = np.reshape(state, [1, STATE_SIZE])
        total = 0.0
        done = False
        while not done:
            q = agent.model.predict(state, verbose=0)
            action = int(np.argmax(q[0]))
            state, reward, terminated, truncated, _ = env.step(action)
            state = np.reshape(state, [1, STATE_SIZE])
            done = terminated or truncated
            total += reward
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards)), rewards


if __name__ == '__main__':
    print('=== FINAL TUNED RUN: 24/24, no target net, seed=42 ===', flush=True)
    agent = DQNAgent(hidden_width=24, use_target=False, seed=42)
    rewards, solved_at, final_avg, elapsed = train_dqn(agent)
    print(f"\ntraining: solved_at={solved_at}, final_avg={final_avg:.2f}, "
          f"elapsed={elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    print('\n=== 100-EPISODE GREEDY EVAL ===', flush=True)
    eval_mean, eval_std, eval_rewards = evaluate(agent, n_episodes=100)
    print(f"greedy eval over 100 eps: {eval_mean:.2f} ± {eval_std:.2f} "
          f"({'PASS' if eval_mean >= SOLVED_THRESHOLD else 'FAIL'} vs {SOLVED_THRESHOLD})",
          flush=True)

    # Save the agent's weights so the notebook can pick them up later.
    agent.model.save_weights('final_tuned_dqn.weights.h5')
    np.savez('final_tuned_results.npz',
             rewards=np.array(rewards),
             eval_rewards=np.array(eval_rewards),
             solved_at=solved_at if solved_at is not None else -1,
             final_avg=final_avg,
             eval_mean=eval_mean,
             eval_std=eval_std,
             elapsed_s=elapsed)
    print('\nSaved: final_tuned_dqn.weights.h5, final_tuned_results.npz', flush=True)
