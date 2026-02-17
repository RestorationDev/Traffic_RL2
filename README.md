Reinforcement Learning for Traffic Signal Control

Overview

This project explores reinforcement learning (RL)–based traffic signal control as a solution to reduce congestion and improve traffic flow in urban networks. Using a microscopic traffic simulator, the system learns adaptive signal policies that minimize cumulative vehicle travel time across the network.

The project progresses from a baseline Deep Q-Network (DQN) to a more stable and effective agent through careful state design, reward engineering, and training improvements.

⸻

Motivation

Traditional traffic signal timing relies on fixed schedules or limited heuristics, which struggle under dynamic traffic conditions. Reinforcement learning offers a data-driven alternative that can:
	•	Adapt to changing traffic patterns
	•	Optimize long-term performance rather than local rules
	•	Scale to complex traffic networks

⸻

Environment
	•	Simulator: CityFlow
	•	Setting: Multi-intersection traffic network
	•	Agent Scope: Single-intersection control (with extensibility to multi-agent setups)

⸻

Methodology

State Space

The agent observes a detailed traffic state constructed from vehicle-level and lane-level information, including:
	•	Vehicle counts per lane
	•	Queue lengths
	•	Waiting times
	•	Traffic pressure metrics

This representation captures both local congestion and network-level imbalance.

⸻

Action Space
	•	Discrete traffic signal phase selections
	•	Enforces realistic signal constraints (e.g., fixed phase sets)

⸻

Reward Function

A custom reward function was designed to encourage global efficiency:
	•	Penalizes large-scale vehicle pressure
	•	Penalizes excessive waiting time
	•	Encourages smoother traffic flow rather than greedy local optimization

This reward structure was critical for learning stable and meaningful policies.

⸻

Learning Algorithm
	•	Baseline: Deep Q-Network (DQN)
	•	Enhancements:
	•	Improved state normalization
	•	Reward shaping for stability
	•	Training and hyperparameter tuning
	•	Experience replay and target network stabilization

⸻

Results
	•	Reduced cumulative travel time compared to fixed-time baselines
	•	Improved queue balance across approaches
	•	More stable signal behavior after reward and training refinements

⸻

Key Takeaways
	•	Careful state and reward design is more impactful than algorithm complexity
	•	Traffic signal control is highly sensitive to reward shaping
	•	RL agents can learn effective policies even with partial observability

⸻

Future Work
	•	Multi-agent RL for network-wide coordination
	•	PPO or other policy-gradient methods
	•	Curriculum learning for large-scale networks
	•	Real-world data integration

⸻

Technologies Used
	•	Python
	•	CityFlow
	•	PyTorch (DQN implementation)
	•	Reinforcement Learning
