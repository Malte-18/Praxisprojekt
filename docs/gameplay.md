# Gameplay Mechanics

## Turn System

The game uses an **alternating turn system**:

1. **Agent's Turn**: The agent chooses and executes an action
2. **Ghost's Turn**: The ghost automatically moves toward the agent

Each call to `interface.step(action)` processes both turns and returns the resulting state.

## Actions

Both the agent and ghost use the same action space:

| Action | Direction | Effect |
|--------|-----------|--------|
| 0 | Left | Move to column - 1 |
| 1 | Down | Move to row + 1 |
| 2 | Right | Move to column + 1 |
| 3 | Up | Move to row - 1 |

**Note**: Row 0 is at the top, so "Down" increases the row number.

## Movement Rules

### Valid Movement

A move is valid if:
1. The destination cell is within grid bounds (4x5)
2. No wall blocks the path between current and destination cells

### Invalid Movement

If a move is invalid, the entity stays in place. No penalty is applied for attempting an invalid move (beyond the standard step penalty).

## Slip Probability

The environment includes stochastic slip mechanics for the agent:

- When the agent attempts to move, there's a configurable probability of **slipping**
- When slipping, the agent moves in a **perpendicular direction** instead:
  - Left/Right → may slip to Up or Down
  - Up/Down → may slip to Left or Right

### Example

With `slip_probability=0.2`:
- 80% chance: Agent moves in intended direction
- 20% chance: Agent slips to one of the two perpendicular directions (randomly chosen)

The slip is reported in the `info` dictionary:

```python
obs, reward, done, info = interface.step(action)
if info.get('slipped'):
    print(f"Intended: {info['intended_action']}, Actual: {info['actual_action']}")
```

## Ghost Behavior

The ghost uses the `ChaseGhostAgent` strategy by default:

1. Calculate the direction toward the agent (row and column differences)
2. Prioritize the axis with the larger distance
3. Try to move in that direction if accessible
4. If blocked, try the other axis
5. If both blocked, try any accessible direction

### Custom Ghost Agents

You can provide a custom ghost agent class:

```python
class MyGhostAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation):
        # observation contains 'agent_relative_pos' and 'neighbors'
        return 0  # Your logic here

interface = AgentInterface(ghost_agent_class=MyGhostAgent)
```

## Terminal States

The episode ends when either:

1. **Agent reaches a goal**: +100 reward, `info['reached_goal'] = True`
2. **Ghost catches agent**: -50 reward, `info['caught_by_ghost'] = True`

The ghost catches the agent when both occupy the same cell after the ghost's move.

## Episode Flow

```
┌─────────────┐
│   reset()   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Agent Turn  │──────────────────────────────────┐
│  (action)   │                                  │
└──────┬──────┘                                  │
       │                                         │
       ▼                                         │
┌─────────────┐         ┌──────────────┐        │
│ Check Goal  │───Yes──▶│  Episode End │        │
└──────┬──────┘         │  (reward +100)│        │
       │ No             └──────────────┘        │
       ▼                                         │
┌─────────────┐                                  │
│ Ghost Turn  │                                  │
│ (automatic) │                                  │
└──────┬──────┘                                  │
       │                                         │
       ▼                                         │
┌─────────────┐         ┌──────────────┐        │
│ Check Catch │───Yes──▶│  Episode End │        │
└──────┬──────┘         │  (reward -50) │        │
       │ No             └──────────────┘        │
       │                                         │
       └─────────────────────────────────────────┘
```

## Rendering

When `render=True`, the environment displays:
- The grid with all cells, colours, and items
- The agent (robot) and ghost positions
- Walls between cells
- An info panel showing:
  - Current step count
  - Agent and ghost positions
  - Manhattan distance between them
  - Current turn indicator
  - Current cell's colour and items
  - Whether current cell is a goal
