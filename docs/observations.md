# Observation Space

## Overview

The observation is a nested dictionary containing information about:
1. The current cell the agent occupies
2. The four neighboring cells (accessibility and colours)
3. The ghost's position relative to the agent

## Observation Structure

```python
{
    "current_cell": {
        "colour": int,          # 0=none, 1=red, 2=green
        "has_item": np.array,   # [dog, flower, notes] - binary array
        "is_goal": int,         # 0 or 1
        "text": str,            # Cell text label (e.g., "Start", "Ziel")
    },
    "neighbors": {
        "up": {
            "accessible": int,   # 0 or 1
            "colour": int,       # 0=none, 1=red, 2=green
        },
        "right": { ... },
        "down": { ... },
        "left": { ... },
    },
    "ghost_relative_pos": np.array,  # [row_diff, col_diff]
}
```

## Detailed Field Descriptions

### Current Cell

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `colour` | int | 0, 1, 2 | 0=none (white), 1=red, 2=green |
| `has_item` | np.array(3) | [0,1] each | Binary flags: [has_dog, has_flower, has_notes] |
| `is_goal` | int | 0, 1 | Whether this cell is a goal cell |
| `text` | str | max 10 chars | Cell text label (e.g., "Start", "Ziel", "") |

### Neighbors

For each direction (`up`, `right`, `down`, `left`):

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `accessible` | int | 0, 1 | Whether movement in this direction is valid (not blocked by wall or boundary) |
| `colour` | int | 0, 1, 2 | Colour of that neighboring cell (0 if out of bounds) |

### Ghost Relative Position

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `ghost_relative_pos` | np.array(2) | [-4, 4] | `[row_diff, col_diff]` from agent to ghost |

**Interpretation:**
- `row_diff > 0`: Ghost is below the agent
- `row_diff < 0`: Ghost is above the agent
- `col_diff > 0`: Ghost is to the right of the agent
- `col_diff < 0`: Ghost is to the left of the agent

## Gymnasium Space Definition

```python
observation_space = gym.spaces.Dict({
    "current_cell": gym.spaces.Dict({
        "colour": gym.spaces.Discrete(3),
        "has_item": gym.spaces.MultiBinary(3),
        "is_goal": gym.spaces.Discrete(2),
        "text": gym.spaces.Text(max_length=10),
    }),
    "neighbors": gym.spaces.Dict({
        "up": gym.spaces.Dict({
            "accessible": gym.spaces.Discrete(2),
            "colour": gym.spaces.Discrete(3),
        }),
        "right": gym.spaces.Dict({...}),
        "down": gym.spaces.Dict({...}),
        "left": gym.spaces.Dict({...}),
    }),
    "ghost_relative_pos": gym.spaces.Box(low=-4, high=4, shape=(2,), dtype=np.int32),
})
```

## Example Observation

```python
{
    "current_cell": {
        "colour": 1,                      # Red coloured cell
        "has_item": np.array([0, 0, 0]),  # No items
        "is_goal": 0,                     # Not a goal
        "text": "",                       # No text label
    },
    "neighbors": {
        "up": {"accessible": 1, "colour": 0},    # Can move up, no colour
        "right": {"accessible": 0, "colour": 2}, # Blocked by wall, green colour
        "down": {"accessible": 1, "colour": 0},  # Can move down, no colour
        "left": {"accessible": 0, "colour": 0},  # At boundary
    },
    "ghost_relative_pos": np.array([-2, 3]),  # Ghost is 2 rows up, 3 columns right
}
```

## Using Observations

### Checking Accessible Directions

```python
def get_valid_actions(obs):
    """Return list of valid action indices."""
    valid = []
    neighbors = obs['neighbors']
    if neighbors['left']['accessible']:
        valid.append(0)
    if neighbors['down']['accessible']:
        valid.append(1)
    if neighbors['right']['accessible']:
        valid.append(2)
    if neighbors['up']['accessible']:
        valid.append(3)
    return valid
```

### Calculating Distance to Ghost

```python
def manhattan_distance_to_ghost(obs):
    """Return Manhattan distance to ghost."""
    rel_pos = obs['ghost_relative_pos']
    return abs(rel_pos[0]) + abs(rel_pos[1])
```

### Checking for Goal

```python
def is_on_goal(obs):
    """Check if agent is on a goal cell."""
    return obs['current_cell']['is_goal'] == 1
```

### Getting Cell Colour

```python
def get_cell_colour(obs, direction=None):
    """Get colour of current cell or neighbor."""
    if direction is None:
        return obs['current_cell']['colour']
    return obs['neighbors'][direction]['colour']
```

### Getting Cell Text

```python
def get_cell_text(obs):
    """Get text label of current cell."""
    return obs['current_cell']['text']
```
