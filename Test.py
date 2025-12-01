import gymnasium as gym
import numpy as np

class CustomGridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self):
        super().__init__()
        self.rows = 4
        self.cols = 5
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Discrete(self.rows * self.cols),
            "ghost": gym.spaces.Discrete(self.rows * self.cols)
        })
        self.action_space = gym.spaces.Discrete(4)  # 0: left, 1: down, 2: right, 3: up

        # 0: empty, 1: obstacle, 2: start, 3: goal, 4+: special
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self._setup_grid()
        self._setup_walls()
        self.agent_pos = [3, 0]
        self.start_pos = [3, 0]
        self.ghost_pos = [1, 2]  # Ghost starts at position (1, 2)
        self.ghost_start_pos = [1, 2]
        self.step_count = 0

    def _setup_grid(self):
        # Fill in grid based on the image (4x5 grid)
        # Row 0: dog, red obstacle, start, empty, dog with music
        # Row 1: empty, green obstacle, ghost, flower, green obstacle
        # Row 2: red obstacle, green obstacle, music note, red obstacle, empty
        # Row 3: robot on red obstacle, ziel, green obstacle with music, flowers, ziel
        
        # 0: empty, 1: red obstacle, 2: green obstacle, 3: start, 4: goal, 5+: special items
        self.grid[0, 0] = 5  # dog
        self.grid[0, 1] = 1  # red obstacle
        self.grid[0, 2] = 3  # start
        self.grid[0, 3] = 0  # empty
        self.grid[0, 4] = 5  # dog with music
        
        self.grid[1, 0] = 0  # empty
        self.grid[1, 1] = 2  # green obstacle
        self.grid[1, 2] = 0  # ghost (agent starts here)
        self.grid[1, 3] = 5  # flower
        self.grid[1, 4] = 2  # green obstacle
        
        self.grid[2, 0] = 1  # red obstacle
        self.grid[2, 1] = 2  # green obstacle
        self.grid[2, 2] = 5  # music note
        self.grid[2, 3] = 1  # red obstacle
        self.grid[2, 4] = 0  # empty
        
        self.grid[3, 0] = 1  # robot on red obstacle
        self.grid[3, 1] = 4  # ziel (goal)
        self.grid[3, 2] = 2  # green obstacle with music
        self.grid[3, 3] = 5  # flowers
        self.grid[3, 4] = 4  # ziel (goal)
    
    def _setup_walls(self):
        
        self.walls_horizontal = np.zeros((self.rows, self.cols), dtype=bool)
        self.walls_vertical = np.zeros((self.rows, self.cols), dtype=bool)

        self.walls_horizontal[0, 3] = True
        self.walls_horizontal[1, 2] = True
        self.walls_horizontal[1, 3] = True
        self.walls_horizontal[1, 3] = True
        self.walls_horizontal[2, 3] = True
        

        self.walls_vertical[0, 0] = True
        self.walls_vertical[1, 2] = True
        self.walls_vertical[2, 1] = True
        self.walls_vertical[2, 2] = True
        self.walls_vertical[3, 1] = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start_pos)
        self.ghost_pos = list(self.ghost_start_pos)
        self.step_count = 0
        return self._get_obs(), {}

    def _move_entity(self, current_pos, action):
        """Helper function to move an entity (agent or ghost) with wall checking"""
        row, col = current_pos
        new_row, new_col = row, col

        if action == 0:
            new_col = col - 1
        elif action == 1:
            new_row = row + 1
        elif action == 2:
            new_col = col + 1
        elif action == 3:
            new_row = row - 1
        

        move_valid = True
        

        if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
            move_valid = False
        

        if move_valid:
            if action == 0:
                if col > 0 and self.walls_vertical[new_row, new_col]:
                    move_valid = False
            elif action == 1:
                if row < self.rows and self.walls_horizontal[new_row - 1, new_col]:
                    move_valid = False
            elif action == 2:
                if col < self.cols and self.walls_vertical[new_row, new_col - 1]:
                    move_valid = False
            elif action == 3:
                if row > 0 and self.walls_horizontal[new_row, new_col]:
                    move_valid = False
        

        if move_valid:
            return [new_row, new_col]
        else:
            return current_pos
    
    def _move_ghost(self):
        """Move the ghost using a simple AI strategy"""

        ghost_row, ghost_col = self.ghost_pos
        agent_row, agent_col = self.agent_pos
        

        possible_actions = []
        

        if agent_row > ghost_row:
            possible_actions.append(1)
        elif agent_row < ghost_row:
            possible_actions.append(3)
        
        if agent_col > ghost_col:
            possible_actions.append(2)
        elif agent_col < ghost_col:
            possible_actions.append(0)
        

        if not possible_actions:
            possible_actions = [0, 1, 2, 3]
        

        for action in possible_actions:
            new_pos = self._move_entity(self.ghost_pos, action)
            if new_pos != self.ghost_pos:
                self.ghost_pos = new_pos
                return
        

        for action in self.np_random.permutation(4):
            new_pos = self._move_entity(self.ghost_pos, action)
            if new_pos != self.ghost_pos:
                self.ghost_pos = new_pos
                return
    
    def step(self, action):
        # Actions: 0: left, 1: down, 2: right, 3: up
        self.step_count += 1
        
        # Move the agent
        self.agent_pos = self._move_entity(self.agent_pos, action)
        
        # Move the ghost (every step)
        self._move_ghost()
        
        # Check if ghost caught the agent
        if self.agent_pos == self.ghost_pos:
            return self._get_obs(), -50, True, False, {"caught_by_ghost": True}
        
        # Calculate reward and check if episode is done
        cell_type = self.grid[self.agent_pos[0], self.agent_pos[1]]
        reward = -1  # Default step penalty
        terminated = False
        
        # TODO: Revise reward system
        if cell_type == 1:  # Red obstacle
            reward = -10
        elif cell_type == 2:  # Green obstacle
            reward = -5
        elif cell_type == 4:  # Goal
            reward = 100
            terminated = True
        
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        # Return both agent and ghost positions
        return {
            "agent": self.agent_pos[0] * self.cols + self.agent_pos[1],
            "ghost": self.ghost_pos[0] * self.cols + self.ghost_pos[1]
        }

    def render(self, mode="human"):
        # ANSI color codes for better visualization
        RESET = "\033[0m"
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"
        WHITE = "\033[97m"
        BG_RED = "\033[101m"
        BG_GREEN = "\033[102m"
        BG_YELLOW = "\033[103m"
        BG_BLUE = "\033[104m"
        BG_MAGENTA = "\033[105m"
        BOLD = "\033[1m"
        
        # Grid symbols with colors
        grid_display = {
            0: f"{WHITE}   {RESET}",  # empty
            1: f"{BG_RED}{WHITE} X {RESET}",  # red obstacle
            2: f"{BG_GREEN}{WHITE} O {RESET}",  # green obstacle
            3: f"{BG_BLUE}{WHITE} S {RESET}",  # start
            4: f"{BG_YELLOW}{WHITE} G {RESET}",  # goal
            5: f"{CYAN} * {RESET}",  # special item
        }
        
        print("\n" + BOLD + "="*50 + RESET)
        print(BOLD + "Custom Grid Environment (4x5) - Step: " + str(self.step_count) + RESET)
        print(BOLD + "="*50 + RESET)
        
        # Top border
        print("┌" + "───┬" * (self.cols - 1) + "───┐")
        
        for row in range(self.rows):
            # Cell content line
            line = "│"
            for col in range(self.cols):
                # Check if agent and ghost are at the same position
                if [row, col] == self.agent_pos and [row, col] == self.ghost_pos:
                    line += f"{BG_RED}{WHITE}{BOLD} ! {RESET}"  # Collision!
                elif [row, col] == self.agent_pos:
                    line += f"{MAGENTA}{BOLD} A {RESET}"  # Agent
                elif [row, col] == self.ghost_pos:
                    line += f"{BG_MAGENTA}{WHITE} 👻{RESET}"  # Ghost
                else:
                    line += grid_display[self.grid[row, col]]
                
                # Show vertical wall or separator
                if col < self.cols - 1:
                    if self.walls_vertical[row, col]:
                        line += f"{BOLD}║{RESET}"  # Thick wall
                    else:
                        line += "│"  # Normal separator
                else:
                    line += "│"
            print(line)
            
            # Horizontal walls/separators
            if row < self.rows - 1:
                wall_line = "├"
                for col in range(self.cols):
                    if self.walls_horizontal[row, col]:
                        wall_line += f"{BOLD}═══{RESET}"  # Thick wall
                    else:
                        wall_line += "───"  # Normal separator
                    
                    if col < self.cols - 1:
                        # Check if there's a cross section with vertical walls
                        has_vertical_here = self.walls_vertical[row, col]
                        has_vertical_below = self.walls_vertical[row + 1, col]
                        has_horizontal_left = col > 0 and self.walls_horizontal[row, col - 1]
                        has_horizontal_right = self.walls_horizontal[row, col]
                        
                        if has_vertical_here or has_vertical_below:
                            wall_line += f"{BOLD}╬{RESET}"
                        else:
                            wall_line += "┼"
                    else:
                        wall_line += "┤"
                print(wall_line)
        
        # Bottom border
        print("└" + "───┴" * (self.cols - 1) + "───┘")
        
        # Legend
        print(f"\n{BOLD}Legend:{RESET}")
        print(f"{MAGENTA}{BOLD} A {RESET} = Agent  {BG_MAGENTA}{WHITE} 👻{RESET} = Ghost (Opposing Agent)  {BG_BLUE}{WHITE} S {RESET} = Start  {BG_YELLOW}{WHITE} G {RESET} = Goal")
        print(f"{BG_RED}{WHITE} X {RESET} = Red Obstacle (-10)  {BG_GREEN}{WHITE} O {RESET} = Green Obstacle (-5)  {CYAN} * {RESET} = Special Item")
        print(f"{BOLD}║{RESET} or {BOLD}═{RESET} = Walls (cannot pass through)")
        print(f"\nAgent Position: ({self.agent_pos[0]}, {self.agent_pos[1]})  |  Ghost Position: ({self.ghost_pos[0]}, {self.ghost_pos[1]})")
        
        # Calculate distance between agent and ghost
        distance = abs(self.agent_pos[0] - self.ghost_pos[0]) + abs(self.agent_pos[1] - self.ghost_pos[1])
        print(f"Distance to Ghost: {distance} (Manhattan distance)")
        print(BOLD + "="*50 + RESET)

# Register the environment
gym.envs.registration.register(
    id="CustomGrid-v0",
    entry_point="your_module:CustomGridEnv",
)

if __name__ == "__main__":
    env = CustomGridEnv()
    obs, info = env.reset()
    
    print("Initial State:")
    env.render()
    
    # Simulate a few moves to show the ghost chasing the agent
    actions = [3, 2, 1, 0]
    action_names = ["up", "right", "down", "left"]
    
    for i, (action, action_name) in enumerate(zip(actions, action_names)):
        print(f"\n\nMove {i+1}: Agent moves {action_name}")
        input("Press Enter to continue...")
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        print(f"Reward: {reward}")
        
        if terminated:
            if info.get("caught_by_ghost", False):
                print("\n🚨 GAME OVER! The ghost caught you! 🚨")
            else:
                print("\n🎉 Congratulations! You reached the goal! 🎉")
            break
    
    print("\n\nSimulation complete!")
