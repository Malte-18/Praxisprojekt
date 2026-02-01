# Reward Structure

## Overview

The reward system encourages the agent to:
1. Reach a goal cell as quickly as possible
2. Avoid being caught by the ghost

## Reward Values

| Event | Reward | Terminal |
|-------|--------|----------|
| Each step (default) | -1 | No |
| Caught by ghost | -50 | **Yes** |
| Reached goal | +100 | **Yes** |

## Reward Calculation Priority

Rewards are calculated in priority order:

1. **Caught by ghost** (highest priority, terminal)
2. **Reached goal** (terminal)
3. **Step penalty** (default)

Only one reward is applied per agent turn.

## Terminal States

The episode ends when either:

| State | Reward | `info` Key |
|-------|--------|------------|
| Agent reaches goal | +100 | `reached_goal: True` |
| Ghost catches agent | -50 | `caught_by_ghost: True` |

## Getting Reward Structure Programmatically

```python
interface = AgentInterface()
rewards = interface.get_reward_structure()
print(rewards)
```

Output:
```python
{
    "step_penalty": -1,
    "caught_by_ghost": -50,
    "reached_goal": 100,
    "slip_probability": 0.2,
    "terminal_states": ["caught_by_ghost", "reached_goal"]
}
```

## Reward Timing

### Agent's Turn

After the agent moves:
- If agent lands on goal → +100 (episode ends)
- Otherwise → -1 (step penalty)

### Ghost's Turn

After the ghost moves:
- If ghost catches agent → -50 (episode ends)
- Otherwise → no additional reward

## Example Episode Rewards

Consider this sequence:

| Step | Agent Action | Result | Reward | Cumulative |
|------|--------------|--------|--------|------------|
| 1 | Move right | Normal cell | -1 | -1 |
| 2 | Move right | Normal cell | -1 | -2 |
| 3 | Move down | Normal cell | -1 | -3 |
| 4 | Move down | Normal cell | -1 | -4 |
| 5 | Move right | **Reached goal** | +100 | **+96** |

## Strategy Implications

### Optimal Play Considerations

1. **Shortest path is best**: Since there are no obstacle penalties, take the most direct route
2. **Ghost avoidance is critical**: The -50 penalty makes survival essential
3. **Speed matters**: Each step costs -1, so minimize total steps

### Risk Assessment

With slip probability of 0.2:
- Each step has a 20% chance of going perpendicular
- This may cause unintended movements toward the ghost
- Factor this risk into pathfinding decisions
- Consider leaving buffer space from walls when possible

## Reward Summary Table

| Condition | Reward | Notes |
|-----------|--------|-------|
| Default step | -1 | Applied every turn |
| Goal reached | +100 | Episode ends |
| Caught by ghost | -50 | Episode ends |
