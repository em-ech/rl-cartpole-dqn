import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import random
from collections import deque

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# Mixed-precision intentionally NOT enabled (audit finding #3).

print('gymnasium', gym.__version__)
print('tensorflow', tf.__version__)
print('numpy', np.__version__)

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
        # Seeding (best-effort; Keras GPU ops are not fully deterministic)
        self.rng = np.random.default_rng(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Network(s)
        self.model = self._build_model(hidden_width, lr)
        if use_target:
            self.target_model = self._build_model(hidden_width, lr)
            self.target_model.set_weights(self.model.get_weights())

    def _build_model(self, hidden_width, lr):
        inputs = Input(shape=(STATE_SIZE,))
        x = Dense(hidden_width, activation='relu')(inputs)
        x = Dense(hidden_width, activation='relu')(x)
        outputs = Dense(ACTION_SIZE, activation='linear')(x)
        m = Model(inputs=inputs, outputs=outputs)
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
        # Vectorisation — the fix the PDF warns about.
        states_arr      = np.vstack(states)
        next_states_arr = np.vstack(next_states)
        rewards_arr     = np.asarray(rewards, dtype=np.float32)
        dones_arr       = np.asarray(dones, dtype=np.float32)
        actions_arr     = np.asarray(actions, dtype=np.int64)

        target_qs = self.model.predict(states_arr, verbose=0)
        bootstrap_model = self.target_model if self.use_target else self.model
        next_qs = bootstrap_model.predict(next_states_arr, verbose=0)

        target_qs[np.arange(self.batch_size), actions_arr] = (
            rewards_arr + self.gamma * np.max(next_qs, axis=1) * (1.0 - dones_arr)
        )
        self.model.fit(states_arr, target_qs, epochs=1, verbose=0)
        self.train_steps += 1
        if self.use_target and self.train_steps % self.target_update_C == 0:
            self.target_model.set_weights(self.model.get_weights())

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


def train_dqn(agent, max_episodes=400, max_steps=500, replay_every=4,
              solved_threshold=SOLVED_THRESHOLD, solved_window=SOLVED_WINDOW,
              verbose=True):
    env = gym.make('CartPole-v1')
    rewards = []
    solved_at = None
    step_counter = 0
    t0 = time.perf_counter()
    for ep in range(max_episodes):
        state, _ = env.reset(seed=int(agent.rng.integers(0, 2**31)))
        state = np.reshape(state, [1, STATE_SIZE])
        total = 0.0
        for step in range(max_steps):
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
        if verbose and (ep + 1) % 20 == 0:
            roll = np.mean(rewards[-min(SOLVED_WINDOW, len(rewards)):])
            print(f"  ep {ep + 1:3d} | last20_avg={np.mean(rewards[-20:]):6.1f} | "
                  f"roll100={roll:6.1f} | eps={agent.eps:.3f}")
        if solved_at is None and len(rewards) >= solved_window:
            if np.mean(rewards[-solved_window:]) >= solved_threshold:
                solved_at = ep + 1
                if verbose:
                    print(f"  SOLVED at ep {solved_at}")
                break
    elapsed = time.perf_counter() - t0
    env.close()
    final_avg = (float(np.mean(rewards[-solved_window:]))
                 if len(rewards) >= solved_window else float(np.mean(rewards)))
    return {'rewards': rewards, 'solved_at': solved_at, 'final_avg': final_avg,
            'elapsed_s': elapsed, 'agent': agent}


def evaluate_dqn(agent, n_episodes=100, seed=12345):
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


def summarise(runs, label):
    solved = [r['solved_at'] if r['solved_at'] is not None else np.nan for r in runs]
    finals = [r['final_avg'] for r in runs]
    elapsed = [r['elapsed_s'] for r in runs]
    return {
        'config': label,
        'solved_at_mean': float(np.nanmean(solved)) if np.any(~np.isnan(solved)) else np.nan,
        'solved_at_std': float(np.nanstd(solved)) if np.any(~np.isnan(solved)) else np.nan,
        'solved_count': int(sum(1 for s in solved if not np.isnan(s))),
        'final_avg_mean': float(np.mean(finals)),
        'final_avg_std': float(np.std(finals)),
        'elapsed_s_mean': float(np.mean(elapsed)),
    }

print('=== sanity baseline: 24/24, no target net, seed=0 ===')
baseline_agent = DQNAgent(hidden_width=24, use_target=False, seed=0)
baseline = train_dqn(baseline_agent, max_episodes=400)
print()
print(f"solved_at: {baseline['solved_at']}")
print(f"final 100-ep avg: {baseline['final_avg']:.2f}")
print(f"elapsed: {baseline['elapsed_s']:.1f}s ({baseline['elapsed_s']/60:.1f} min)")
print(f"episodes run: {len(baseline['rewards'])}")

fig, ax = plt.subplots(figsize=(10, 4))
r = np.array(baseline['rewards'])
roll = pd.Series(r).rolling(SOLVED_WINDOW, min_periods=1).mean()
ax.plot(r, alpha=0.3, label='episode reward')
ax.plot(roll, label=f'{SOLVED_WINDOW}-episode rolling mean', linewidth=2)
ax.axhline(SOLVED_THRESHOLD, color='red', linestyle='--', label=f'solved ({SOLVED_THRESHOLD})')
ax.set_xlabel('episode')
ax.set_ylabel('reward')
ax.set_title('Sanity baseline — audited DQN (24/24, no target net)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

WIDTHS = [16, 24, 64]
width_results = {}
width_summaries = []
for w in WIDTHS:
    label = f'{w}/{w}'
    runs = []
    for s in range(2):
        print(f'=== width sweep: {label}, seed={s} ===')
        agent = DQNAgent(hidden_width=w, use_target=False, seed=s)
        runs.append(train_dqn(agent, max_episodes=400, verbose=False))
        print(f"  -> solved_at={runs[-1]['solved_at']}, final_avg={runs[-1]['final_avg']:.1f}, "
              f"elapsed={runs[-1]['elapsed_s']:.0f}s")
    width_results[label] = runs
    width_summaries.append(summarise(runs, label))
    s_ = width_summaries[-1]
    print(f"  SUMMARY {label}: solved@{s_['solved_at_mean']:.1f}±{s_['solved_at_std']:.1f} "
          f"final={s_['final_avg_mean']:.1f}±{s_['final_avg_std']:.1f} ({s_['solved_count']}/2)")
    print()

width_df = pd.DataFrame(width_summaries)
width_df

fig, ax = plt.subplots(figsize=(10, 4))
for label, runs in width_results.items():
    max_len = max(len(r['rewards']) for r in runs)
    padded = np.full((len(runs), max_len), np.nan)
    for i, r in enumerate(runs):
        padded[i, :len(r['rewards'])] = r['rewards']
    roll = pd.Series(np.nanmean(padded, axis=0)).rolling(SOLVED_WINDOW, min_periods=1).mean()
    ax.plot(roll, label=label, linewidth=2)
ax.axhline(SOLVED_THRESHOLD, color='red', linestyle='--', label='solved')
ax.set_xlabel('episode')
ax.set_ylabel(f'{SOLVED_WINDOW}-ep rolling mean reward (2-seed avg)')
ax.set_title('Network width sweep')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Pick best width by combining solved-count then solved-speed then final-avg
width_sorted = width_df.sort_values(['solved_count', 'solved_at_mean', 'final_avg_mean'],
                                     ascending=[False, True, False])
BEST_WIDTH = int(width_sorted.iloc[0]['config'].split('/')[0])
print(f"best width from Experiment 1: {BEST_WIDTH}/{BEST_WIDTH}")

TARGET_CONFIGS = {
    'no target net':       dict(use_target=False),
    'target net (C=200)':  dict(use_target=True, target_update_C=200),
}

target_results = {}
target_summaries = []
for label, kw in TARGET_CONFIGS.items():
    runs = []
    for s in range(2):
        print(f'=== target sweep: {label}, seed={s} ===')
        agent = DQNAgent(hidden_width=BEST_WIDTH, seed=s, **kw)
        runs.append(train_dqn(agent, max_episodes=400, verbose=False))
        print(f"  -> solved_at={runs[-1]['solved_at']}, final_avg={runs[-1]['final_avg']:.1f}, "
              f"elapsed={runs[-1]['elapsed_s']:.0f}s")
    target_results[label] = runs
    target_summaries.append(summarise(runs, label))
    s_ = target_summaries[-1]
    print(f"  SUMMARY {label}: solved@{s_['solved_at_mean']:.1f}±{s_['solved_at_std']:.1f} "
          f"final={s_['final_avg_mean']:.1f}±{s_['final_avg_std']:.1f} ({s_['solved_count']}/2)")
    print()

target_df = pd.DataFrame(target_summaries)
target_df

fig, ax = plt.subplots(figsize=(10, 4))
for label, runs in target_results.items():
    max_len = max(len(r['rewards']) for r in runs)
    padded = np.full((len(runs), max_len), np.nan)
    for i, r in enumerate(runs):
        padded[i, :len(r['rewards'])] = r['rewards']
    roll = pd.Series(np.nanmean(padded, axis=0)).rolling(SOLVED_WINDOW, min_periods=1).mean()
    ax.plot(roll, label=label, linewidth=2)
ax.axhline(SOLVED_THRESHOLD, color='red', linestyle='--', label='solved')
ax.set_xlabel('episode')
ax.set_ylabel(f'{SOLVED_WINDOW}-ep rolling mean reward (2-seed avg)')
ax.set_title(f'Target network sweep at width {BEST_WIDTH}/{BEST_WIDTH}')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

target_sorted = target_df.sort_values(['solved_count', 'solved_at_mean', 'final_avg_mean'],
                                       ascending=[False, True, False])
BEST_TARGET_LABEL = target_sorted.iloc[0]['config']
BEST_TARGET_KW = TARGET_CONFIGS[BEST_TARGET_LABEL]
print(f"best target config: {BEST_TARGET_LABEL} -> {BEST_TARGET_KW}")
print(f"final config: width={BEST_WIDTH}/{BEST_WIDTH}, {BEST_TARGET_LABEL}")
print()

print('=== final tuned run, seed=42 ===')
final_agent = DQNAgent(hidden_width=BEST_WIDTH, seed=42, **BEST_TARGET_KW)
final = train_dqn(final_agent, max_episodes=400)
print()
print(f"training: solved_at={final['solved_at']}, final_avg={final['final_avg']:.2f}, "
      f"elapsed={final['elapsed_s']:.1f}s ({final['elapsed_s']/60:.1f} min)")

print('=== 100-episode greedy eval ===')
eval_mean, eval_std, eval_rewards = evaluate_dqn(final_agent, n_episodes=100)
print(f"greedy eval over 100 eps: {eval_mean:.2f} ± {eval_std:.2f} "
      f"({'PASS' if eval_mean >= SOLVED_THRESHOLD else 'FAIL'} vs {SOLVED_THRESHOLD})")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
r = np.array(final['rewards'])
roll = pd.Series(r).rolling(SOLVED_WINDOW, min_periods=1).mean()
axes[0].plot(r, alpha=0.3, label='episode reward')
axes[0].plot(roll, label=f'{SOLVED_WINDOW}-ep rolling mean', linewidth=2)
axes[0].axhline(SOLVED_THRESHOLD, color='red', linestyle='--', label='solved')
axes[0].set_xlabel('episode')
axes[0].set_ylabel('reward')
axes[0].set_title('Final tuned DQN training run')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(eval_rewards, bins=20, edgecolor='black')
axes[1].axvline(eval_mean, color='red', linestyle='--', label=f'mean {eval_mean:.1f}')
axes[1].axvline(SOLVED_THRESHOLD, color='black', linestyle=':', label=f'solved {SOLVED_THRESHOLD}')
axes[1].set_xlabel('episode reward')
axes[1].set_ylabel('count')
axes[1].set_title('Greedy eval over 100 episodes')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Raw sweep tables — kept at the bottom for reference / sanity-checking the cited numbers.
print('Experiment 1 — Network width')
print('-' * 60)
print(width_df[['config', 'solved_at_mean', 'solved_at_std', 'solved_count',
                'final_avg_mean', 'final_avg_std', 'elapsed_s_mean']].to_string(index=False))
print()
print('Experiment 2 — Target network')
print('-' * 60)
print(target_df[['config', 'solved_at_mean', 'solved_at_std', 'solved_count',
                 'final_avg_mean', 'final_avg_std', 'elapsed_s_mean']].to_string(index=False))
print()
print('Final tuned run')
print('-' * 60)
print(f"  config: width={BEST_WIDTH}/{BEST_WIDTH}, {BEST_TARGET_LABEL}")
print(f"  episodes to solve: {final['solved_at']}")
print(f"  wall-clock: {final['elapsed_s']:.1f} s")
print(f"  100-episode greedy eval: {eval_mean:.2f} ± {eval_std:.2f}")
