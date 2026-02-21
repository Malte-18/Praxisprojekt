# Environment Structure

## Grid Layout

The environment is a **4x5 grid** (4 rows, 5 columns) with the following coordinate system:

![](Umgebung.png)

**Starting Positions:**
- Agent starts at position `[2, 0]` (row 2, column 0)
- Ghost starts at position `[0, 3]` (row 0, column 3)

## Cell Properties

Each cell contains:

| Property | Type | Description |
|----------|------|-------------|
| `colour` | int | 0 = none (white), 1 = red, 2 = green |
| `items` | list | Items present: `dog`, `flower`, `one_note`, `two_notes` |
| `is_goal` | bool | Whether this cell is a goal |
| `is_start` | bool | Whether this is the start position marker |
| `text` | str | Display text for the cell (e.g., "Start", "Ziel") |

## Cell Colours

Cells have colours that are displayed as crosshatch patterns. These are informational and do not affect rewards.

| Colour Value | Visual | Description |
|--------------|--------|-------------|
| 0 | White cell | No colour pattern |
| 1 | Red crosshatch pattern | Red coloured cell |
| 2 | Green crosshatch pattern | Green coloured cell |

## Items

Items are decorative objects displayed in cells:

| Item | Visual | Description |
|------|--------|-------------|
| `dog` | Gray dog icon | A dog |
| `flower` | White flower with yellow center | A flower |
| `one_note` | Single musical note | A single note |
| `two_notes` | Double musical notes | Two notes |

## Goals

There are **two goal cells**:
- Position `[3, 1]` - Second column, bottom row
- Position `[3, 4]` - Fifth column, bottom row

Reaching either goal ends the episode with a +100 reward.

## Walls

Walls block movement between adjacent cells. The environment has the following walls:

### Horizontal Walls (block vertical movement)

| Location | Blocks movement between |
|----------|------------------------|
| `[0, 3]` | Row 0 and Row 1 at column 3 |
| `[1, 2]` | Row 1 and Row 2 at column 2 |
| `[1, 3]` | Row 1 and Row 2 at column 3 |
| `[2, 3]` | Row 2 and Row 3 at column 3 |

### Vertical Walls (block horizontal movement)

| Location | Blocks movement between |
|----------|------------------------|
| `[0, 0]` | Column 0 and Column 1 at row 0 |
| `[1, 2]` | Column 2 and Column 3 at row 1 |
| `[2, 1]` | Column 1 and Column 2 at row 2 |
| `[2, 2]` | Column 2 and Column 3 at row 2 |
| `[3, 1]` | Column 1 and Column 2 at row 3 |

## Visual Representation

When rendered, the environment displays:

- **Grid**: White cells with gray grid lines
- **Coloured Cells**: Crosshatch patterns (red or green)
- **Agent**: Gray robot with antenna and "GPS" label
- **Ghost**: Cyan Pac-Man style ghost
- **Goals**: Cells labeled "Ziel" (German for "Goal")
- **Start**: Cell labeled "Start"
- **Walls**: Black bars between cells
- **Info Panel**: Bottom panel showing step count, positions, and current cell info
