# API Reference

## AgentInterface

The main interface for interacting with the environment. Handles the agent-ghost turn alternation internally.

### Constructor

```python
AgentInterface(render=True, step_delay=100, slip_probability=0.2, ghost_agent_class=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `render` | bool | `True` | Enable graphical display |
| `step_delay` | int | `100` | Milliseconds between rendered steps |
| `slip_probability` | float | `0.2` | Probability of slipping perpendicular (0.0-1.0) |
| `ghost_agent_class` | class | `None` | Custom ghost agent class (uses `ChaseGhostAgent` if None) |

### Methods

#### `reset(seed=None)`

Reset the environment for a new episode.

```python
obs = interface.reset(seed=42)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | `None` | Random seed for reproducibility |

**Returns:** Initial observation dictionary

---

#### `step(action)`

Execute one full turn (agent + ghost).

```python
obs, reward, done, info = interface.step(action)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `action` | int | Agent action (0=left, 1=down, 2=right, 3=up) |

**Returns:** Tuple of (observation, reward, done, info)

| Return Value | Type | Description |
|--------------|------|-------------|
| `observation` | dict | New observation after both moves |
| `reward` | float | Total reward for this step |
| `done` | bool | Whether episode has ended |
| `info` | dict | Additional information |

**Info dictionary keys:**
- `slipped` (bool): Whether agent slipped
- `intended_action` (str): Intended direction name
- `actual_action` (str): Actual direction after slip
- `reached_goal` (bool): Whether goal was reached
- `caught_by_ghost` (bool): Whether ghost caught agent
- `current_turn` (str): 'agent' or 'ghost'
- `mover` (str): Who just moved

---

#### `is_terminated()`

Check if episode has ended.

```python
if interface.is_terminated():
    print("Episode over!")
```

**Returns:** `bool`

---

#### `get_episode_stats()`

Get statistics for current/last episode.

```python
stats = interface.get_episode_stats()
```

**Returns:** Dictionary with:
- `total_reward` (float): Cumulative reward
- `steps` (int): Number of steps taken
- `terminated` (bool): Episode ended normally
- `truncated` (bool): Episode was truncated
- `reached_goal` (bool): Agent reached goal
- `caught_by_ghost` (bool): Ghost caught agent

---

#### `get_action_space()`

Get the Gymnasium action space.

```python
action_space = interface.get_action_space()
# Returns: Discrete(4)
```

---

#### `get_observation_space()`

Get the Gymnasium observation space.

```python
obs_space = interface.get_observation_space()
```

---

#### `get_reward_structure()`

Get the reward configuration.

```python
rewards = interface.get_reward_structure()
```

**Returns:** Dictionary with reward values

---

#### `close()`

Clean up resources (close Pygame window).

```python
interface.close()
```

---

## CustomGridEnv

The underlying Gymnasium environment. Use `AgentInterface` for typical usage.

### Registration

The environment is registered as `CustomGrid-v0`:

```python
import gymnasium as gym
env = gym.make("CustomGrid-v0")
```

### Direct Usage

For advanced use cases requiring direct environment access:

```python
from Enviroment import CustomGridEnv

env = CustomGridEnv(render_mode="human", slip_probability=0.2)
obs, info = env.reset()

# Manual turn handling
agent_obs, reward, terminated, truncated, info = env.step(agent_action)
ghost_obs = env._get_ghost_obs()
ghost_obs, reward, terminated, truncated, info = env.step(ghost_action)

env.render()
env.close()
```

---

## Agent Classes

### Creating a Custom Agent

```python
class MyAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation):
        """
        Choose an action based on observation.
        
        Args:
            observation: Dict with current_cell (colour, has_item, is_goal, text),
                        neighbors (up/right/down/left with accessible, colour),
                        and ghost_relative_pos
            
        Returns:
            int: Action (0=left, 1=down, 2=right, 3=up)
        """
        # Your logic here
        return 0
```

### Built-in Agents

#### RandomAgent

Chooses random actions.

```python
from Enviroment import RandomAgent

agent = RandomAgent(interface.get_action_space())
action = agent.get_action(obs)
```

#### ChaseGhostAgent

Ghost agent that chases the player (default ghost behavior).

```python
from Enviroment import ChaseGhostAgent

ghost = ChaseGhostAgent(env.action_space)
ghost_action = ghost.get_action(ghost_obs)
```

#### RandomGhostAgent

Ghost agent that moves randomly.

```python
from Enviroment import RandomGhostAgent

interface = AgentInterface(ghost_agent_class=RandomGhostAgent)
```

---

## Utility Functions

### run_with_agent

Run multiple episodes with specified agents.

```python
from Enviroment import run_with_agent, RandomAgent

stats = run_with_agent(
    agent_class=None,           # Your agent class (None = RandomAgent)
    ghost_agent_class=None,     # Ghost class (None = ChaseGhostAgent)
    num_episodes=10,            # Number of episodes
    render=True,                # Show graphics
    step_delay=100,             # Ms between steps
    slip_probability=0.2        # Slip probability
)
```

**Returns:** List of episode statistics dictionaries

---

## Complete Example

```python
from Enviroment import AgentInterface

class SmartAgent:
    """Agent that avoids ghost and finds goal."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, obs):
        # Get ghost direction
        ghost_rel = obs['ghost_relative_pos']
        neighbors = obs['neighbors']
        
        # Prefer directions away from ghost
        best_action = None
        best_score = -float('inf')
        
        directions = [
            (0, 'left', 0, -1),
            (1, 'down', 1, 0),
            (2, 'right', 0, 1),
            (3, 'up', -1, 0)
        ]
        
        for action, name, dr, dc in directions:
            if not neighbors[name]['accessible']:
                continue
            
            score = 0
            # Prefer moving away from ghost
            if dr != 0:
                score += dr * (-ghost_rel[0])
            if dc != 0:
                score += dc * (-ghost_rel[1])
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action is not None else self.action_space.sample()


# Run the agent
interface = AgentInterface(render=True, slip_probability=0.2)

for episode in range(5):
    obs = interface.reset()
    agent = SmartAgent(interface.get_action_space())
    
    while not interface.is_terminated():
        action = agent.get_action(obs)
        obs, reward, done, info = interface.step(action)
    
    stats = interface.get_episode_stats()
    result = "WIN" if stats['reached_goal'] else "LOSE"
    print(f"Episode {episode+1}: {result}, Reward: {stats['total_reward']}")

interface.close()
```
