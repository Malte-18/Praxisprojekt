# CustomGrid Environment Documentation

A Gymnasium-based grid environment featuring an agent navigating a 4x5 grid to reach goal cells while avoiding a chasing ghost.

![Demo Video](Aufzeichnung_Umgebung.mp4)

## Overview

CustomGrid is a turn-based environment where:
- An **agent** (robot with GPS) tries to reach one of the goal cells
- A **ghost** chases the agent each turn
- **Walls** block movement between certain cells
- **Slip probability** adds stochasticity - the agent may slip perpendicular to intended direction
- **Coloured cells** provide visual information (red and green patterns)

## Quick Start

```python
from src.Environment import AgentInterface, RandomAgent

# Create the interface
interface = AgentInterface(render=True, slip_probability=0.2)

# Reset and get initial observation
obs = interface.reset()

# Create your agent
agent = RandomAgent(interface.get_action_space())

# Run an episode
while not interface.is_terminated():
    action = agent.get_action(obs)
    obs, reward, done, info = interface.step(action)

# Get results
stats = interface.get_episode_stats()
print(f"Total reward: {stats['total_reward']}")

interface.close()
```

## Documentation Index

| Document                        | Description                                               |
|---------------------------------|-----------------------------------------------------------|
| [Environment](environment.md)   | Grid layout, cells, colours, and items                    |
| [Gameplay](gameplay.md)         | Turn system, movement, ghost behavior, and slip mechanics |
| [Observations](observations.md) | Observation space structure and contents                  |
| [Rewards](rewards.md)           | Reward structure and terminal states                      |
| [API Reference](api.md)         | Complete API documentation                                |

## Requirements

- Python 3.8+
- gymnasium
- numpy
- pygame

## Installation

```bash
pip install gymnasium numpy pygame
```

## Running the Demo

```bash
python Enviroment.py
```

This runs 3 episodes with a random agent and displays the graphical interface.
