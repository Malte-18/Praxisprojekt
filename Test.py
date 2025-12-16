import gymnasium as gym
import numpy as np

class CustomGridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self):
        super().__init__()
        self.rows = 4
        self.cols = 5
        self.observation_space = gym.spaces.Dict({
            "current_cell": gym.spaces.Dict({
                "obstacle": gym.spaces.Discrete(3),  # 0=none, 1=red, 2=green
                "has_item": gym.spaces.MultiBinary(3),  # [dog, flower, notes]
                "is_goal": gym.spaces.Discrete(2),
            }),
            "neighbors": gym.spaces.Dict({
                "up": gym.spaces.Dict({
                    "accessible": gym.spaces.Discrete(2),
                    "obstacle": gym.spaces.Discrete(3),
                }),
                "right": gym.spaces.Dict({
                    "accessible": gym.spaces.Discrete(2),
                    "obstacle": gym.spaces.Discrete(3),
                }),
                "down": gym.spaces.Dict({
                    "accessible": gym.spaces.Discrete(2),
                    "obstacle": gym.spaces.Discrete(3),
                }),
                "left": gym.spaces.Dict({
                    "accessible": gym.spaces.Discrete(2),
                    "obstacle": gym.spaces.Discrete(3),
                }),
            }),
            "ghost_relative_pos": gym.spaces.Box(low=-4, high=4, shape=(2,), dtype=np.int32),
        })
        
        self.action_space = gym.spaces.Discrete(4)  # 0: left, 1: down, 2: right, 3: up

        # Use object dtype to store complex field information
        self.grid = np.empty((self.rows, self.cols), dtype=object)
        self._setup_grid()
        self._setup_walls()
        self.agent_pos = [3, 0]
        self.start_pos = [3, 0]
        self.ghost_pos = [0, 3]
        self.ghost_start_pos = [0, 3]
        self.step_count = 0

    def _setup_grid(self):
        # Each cell is a dictionary with: obstacle, items, is_goal, is_start
        # obstacle: 0=none, 1=red, 2=green
        # items: list of items (dog, flower, notes)
        
        self.grid[0, 0] = {'obstacle': 2, 'items': ['dog'], 'is_goal': False, 'is_start': False}
        self.grid[0, 1] = {'obstacle': 1, 'items': [], 'is_goal': False, 'is_start': False}
        self.grid[0, 2] = {'obstacle': 0, 'items': [], 'is_goal': False, 'is_start': True}
        self.grid[0, 3] = {'obstacle': 0, 'items': [], 'is_goal': False, 'is_start': False}
        self.grid[0, 4] = {'obstacle': 0, 'items': ['dog', 'one_note'], 'is_goal': False, 'is_start': False}
        
        self.grid[1, 0] = {'obstacle': 0, 'items': [], 'is_goal': False, 'is_start': False}
        self.grid[1, 1] = {'obstacle': 2, 'items': [], 'is_goal': False, 'is_start': False}
        self.grid[1, 2] = {'obstacle': 0, 'items': [], 'is_goal': False, 'is_start': False}
        self.grid[1, 3] = {'obstacle': 0, 'items': ['flower'], 'is_goal': False, 'is_start': False}
        self.grid[1, 4] = {'obstacle': 2, 'items': [], 'is_goal': False, 'is_start': False}
        
        self.grid[2, 0] = {'obstacle': 1, 'items': [], 'is_goal': False, 'is_start': False}
        self.grid[2, 1] = {'obstacle': 2, 'items': [], 'is_goal': False, 'is_start': False}
        self.grid[2, 2] = {'obstacle': 0, 'items': ['one_note'], 'is_goal': False, 'is_start': False}
        self.grid[2, 3] = {'obstacle': 1, 'items': [], 'is_goal': False, 'is_start': False}
        self.grid[2, 4] = {'obstacle': 0, 'items': [], 'is_goal': False, 'is_start': False}
        
        self.grid[3, 0] = {'obstacle': 1, 'items': [], 'is_goal': False, 'is_start': False}
        self.grid[3, 1] = {'obstacle': 0, 'items': [], 'is_goal': True, 'is_start': False}
        self.grid[3, 2] = {'obstacle': 2, 'items': ['two_notes'], 'is_goal': False, 'is_start': False}
        self.grid[3, 3] = {'obstacle': 0, 'items': ['flower'], 'is_goal': False, 'is_start': False}
        self.grid[3, 4] = {'obstacle': 0, 'items': [], 'is_goal': True, 'is_start': False}
    
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
    
    @staticmethod
    def get_reward_structure():
        """
        Get the reward structure for this environment.
        
        Returns:
            dict: Dictionary describing all rewards and their values
        """
        return {
            "step_penalty": -1,
            "red_obstacle": -10,
            "green_obstacle": -5,
            "caught_by_ghost": -50,
            "reached_goal": 100,
            "terminal_states": ["caught_by_ghost", "reached_goal"]
        }
    
    def calculate_reward(self, caught_by_ghost=False):
        """
        Calculate reward based on current game state.
        
        Reward Structure:
        ----------------
        - Caught by ghost: -50 (terminal)
        - Reached goal: +100 (terminal)
        - Red obstacle: -10
        - Green obstacle: -5
        - Default step: -1
        
        Args:
            caught_by_ghost (bool): Whether the ghost caught the agent
            
        Returns:
            tuple: (reward, terminated, info_dict)
        """
        reward_structure = self.get_reward_structure()
        reward = reward_structure["step_penalty"]  # Default step penalty
        terminated = False
        info = {}
        
        # Priority 1: Check if caught by ghost (terminal state)
        if caught_by_ghost:
            return reward_structure["caught_by_ghost"], True, {"caught_by_ghost": True}
        
        # Get current cell properties
        current_cell = self.grid[self.agent_pos[0], self.agent_pos[1]]
        
        # Priority 2: Check if goal reached (terminal state)
        if current_cell['is_goal']:
            return reward_structure["reached_goal"], True, {"reached_goal": True}
        
        # Priority 3: Check obstacle penalties
        if current_cell['obstacle'] == 1:  # Red obstacle
            reward = reward_structure["red_obstacle"]
            info['hit_obstacle'] = 'red'
        elif current_cell['obstacle'] == 2:  # Green obstacle
            reward = reward_structure["green_obstacle"]
            info['hit_obstacle'] = 'green'
        
        return reward, terminated, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int): 0=left, 1=down, 2=right, 3=up
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        
        # Move the agent
        self.agent_pos = self._move_entity(self.agent_pos, action)
        
        # Move the ghost
        self._move_ghost()
        
        # Check if ghost caught the agent
        caught_by_ghost = (self.agent_pos == self.ghost_pos)
        
        # Calculate reward using separate reward function
        reward, terminated, info = self.calculate_reward(caught_by_ghost)
        
        return self._get_obs(), reward, terminated, False, info

    def _get_cell_info(self, row, col):
        """Get information about a specific cell"""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return None
        return self.grid[row, col]
    
    def _is_move_valid(self, from_pos, to_pos):
        """Check if a move from one position to another is valid (not blocked by wall)"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        if to_row < 0 or to_row >= self.rows or to_col < 0 or to_col >= self.cols:
            return False
        
        if to_row < from_row:  
            return not (from_row > 0 and self.walls_horizontal[from_row - 1, from_col])
        elif to_row > from_row:  
            return not (from_row < self.rows and self.walls_horizontal[from_row, from_col])
        elif to_col < from_col:  
            return not (from_col > 0 and self.walls_vertical[from_row, from_col - 1])
        elif to_col > from_col:  
            return not (from_col < self.cols and self.walls_vertical[from_row, from_col])
        
        return True
    
    def _get_obs(self):
        """Return detailed observation about current cell and neighbors"""
        row, col = self.agent_pos
        current_cell = self.grid[row, col]
        
        current_obs = {
            "obstacle": current_cell['obstacle'],
            "has_item": np.array([
                1 if 'dog' in current_cell['items'] else 0,
                1 if 'flower' in current_cell['items'] else 0,
                1 if any(note in current_cell['items'] for note in ['one_note', 'two_notes']) else 0
            ], dtype=np.int8),
            "is_goal": 1 if current_cell['is_goal'] else 0,
        }
        
        neighbors_obs = {}
        
        up_pos = [row - 1, col]
        up_accessible = self._is_move_valid(self.agent_pos, up_pos)
        up_cell = self._get_cell_info(row - 1, col)
        neighbors_obs["up"] = {
            "accessible": 1 if up_accessible else 0,
            "obstacle": up_cell['obstacle'] if up_cell else 0,
        }
        
        right_pos = [row, col + 1]
        right_accessible = self._is_move_valid(self.agent_pos, right_pos)
        right_cell = self._get_cell_info(row, col + 1)
        neighbors_obs["right"] = {
            "accessible": 1 if right_accessible else 0,
            "obstacle": right_cell['obstacle'] if right_cell else 0,
        }
        
        down_pos = [row + 1, col]
        down_accessible = self._is_move_valid(self.agent_pos, down_pos)
        down_cell = self._get_cell_info(row + 1, col)
        neighbors_obs["down"] = {
            "accessible": 1 if down_accessible else 0,
            "obstacle": down_cell['obstacle'] if down_cell else 0,
        }
        
        left_pos = [row, col - 1]
        left_accessible = self._is_move_valid(self.agent_pos, left_pos)
        left_cell = self._get_cell_info(row, col - 1)
        neighbors_obs["left"] = {
            "accessible": 1 if left_accessible else 0,
            "obstacle": left_cell['obstacle'] if left_cell else 0,
        }
        
        ghost_relative = np.array([
            self.ghost_pos[0] - self.agent_pos[0],
            self.ghost_pos[1] - self.agent_pos[1]
        ], dtype=np.int32)
        
        return {
            "current_cell": current_obs,
            "neighbors": neighbors_obs,
            "ghost_relative_pos": ghost_relative,
        }

    def _cell_to_display(self, cell):
        """Convert cell dictionary to display string"""
        RESET = "\033[0m"
        WHITE = "\033[97m"
        BG_RED = "\033[101m"
        BG_GREEN = "\033[102m"
        BG_YELLOW = "\033[103m"
        BG_BLUE = "\033[104m"
        CYAN = "\033[96m"
        
        if cell['is_goal']:
            return f"{BG_YELLOW}{WHITE} G {RESET}"
        elif cell['is_start']:
            return f"{BG_BLUE}{WHITE} S {RESET}"
        elif cell['obstacle'] == 1:  # Red
            return f"{BG_RED}{WHITE} X {RESET}"
        elif cell['obstacle'] == 2:  # Green
            return f"{BG_GREEN}{WHITE} O {RESET}"
        elif cell['items']:
            item = cell['items'][0]
            if 'dog' in item:
                return f"{CYAN} D {RESET}"
            elif 'flower' in item:
                return f"{CYAN} F {RESET}"
            elif 'note' in item:
                return f"{CYAN} ♪ {RESET}"
            else:
                return f"{CYAN} * {RESET}"
        else:
            return f"{WHITE}   {RESET}"
    
    def render(self, mode="human"):
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
        
        print("\n" + BOLD + "="*50 + RESET)
        print(BOLD + "Custom Grid Environment (4x5) - Step: " + str(self.step_count) + RESET)
        print(BOLD + "="*50 + RESET)
        
        print("┌" + "───┬" * (self.cols - 1) + "───┐")
        
        for row in range(self.rows):
            line = "│"
            for col in range(self.cols):
                if [row, col] == self.agent_pos and [row, col] == self.ghost_pos:
                    line += f"{BG_RED}{WHITE}{BOLD} ! {RESET}"  
                elif [row, col] == self.agent_pos:
                    line += f"{MAGENTA}{BOLD} A {RESET}"  
                elif [row, col] == self.ghost_pos:
                    line += f"{BG_MAGENTA}{WHITE} 👻{RESET}"  
                else:
                    line += self._cell_to_display(self.grid[row, col])
                
                if col < self.cols - 1:
                    if self.walls_vertical[row, col]:
                        line += f"{BOLD}║{RESET}"  
                    else:
                        line += "│"  
                else:
                    line += "│"
            print(line)
            
            if row < self.rows - 1:
                wall_line = "├"
                for col in range(self.cols):
                    if self.walls_horizontal[row, col]:
                        wall_line += f"{BOLD}═══{RESET}"  
                    else:
                        wall_line += "───"  
                    
                    if col < self.cols - 1:
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
        
        print("└" + "───┴" * (self.cols - 1) + "───┘")
        
        print(f"\n{BOLD}Legend:{RESET}")
        print(f"{MAGENTA}{BOLD} A {RESET} = Agent  {BG_MAGENTA}{WHITE} 👻{RESET} = Ghost (Opposing Agent)  {BG_BLUE}{WHITE} S {RESET} = Start  {BG_YELLOW}{WHITE} G {RESET} = Goal")
        print(f"{BG_RED}{WHITE} X {RESET} = Red Obstacle (-10)  {BG_GREEN}{WHITE} O {RESET} = Green Obstacle (-5)")
        print(f"{CYAN} D {RESET} = Dog  {CYAN} F {RESET} = Flower  {CYAN} ♪ {RESET} = Notes")
        print(f"{BOLD}║{RESET} or {BOLD}═{RESET} = Walls (cannot pass through)")
        
        current_cell = self.grid[self.agent_pos[0], self.agent_pos[1]]
        print(f"\n{BOLD}Current Cell Details:{RESET}")
        print(f"  Position: ({self.agent_pos[0]}, {self.agent_pos[1]})")
        print(f"  Obstacle: {'Red' if current_cell['obstacle'] == 1 else 'Green' if current_cell['obstacle'] == 2 else 'None'}")
        print(f"  Items: {', '.join(current_cell['items']) if current_cell['items'] else 'None'}")
        print(f"  Is Goal: {'Yes' if current_cell['is_goal'] else 'No'}")
        
        obs = self._get_obs()
        print(f"\n{BOLD}Neighbors:{RESET}")
        for direction in ['up', 'right', 'down', 'left']:
            neighbor = obs['neighbors'][direction]
            accessible = "✓" if neighbor['accessible'] else "✗"
            obstacle = 'Red' if neighbor['obstacle'] == 1 else 'Green' if neighbor['obstacle'] == 2 else 'None'
            print(f"  {direction.capitalize():>5}: {accessible} Accessible | Obstacle: {obstacle}")
        
        print(f"\n{BOLD}Ghost Info:{RESET}")
        print(f"  Position: ({self.ghost_pos[0]}, {self.ghost_pos[1]})")
        print(f"  Relative to Agent: {obs['ghost_relative_pos']}")
        distance = abs(self.agent_pos[0] - self.ghost_pos[0]) + abs(self.agent_pos[1] - self.ghost_pos[1])
        print(f"  Manhattan Distance: {distance}")
        
        print(BOLD + "="*50 + RESET)

gym.envs.registration.register(
    id="CustomGrid-v0",
    entry_point="your_module:CustomGridEnv",
)

if __name__ == "__main__":
    env = CustomGridEnv()
    
    # Display reward structure
    print("\n" + "="*50)
    print("REWARD STRUCTURE:")
    print("="*50)
    rewards = env.get_reward_structure()
    for key, value in rewards.items():
        if key != "terminal_states":
            print(f"  {key.replace('_', ' ').title()}: {value:+d}" if isinstance(value, int) else f"  {key}: {value}")
    print(f"  Terminal States: {', '.join(rewards['terminal_states'])}")
    print("="*50)
    
    obs, info = env.reset()
    
    print("\nInitial State:")
    env.render()
    
    print("\n" + "="*50)
    print("OBSERVATION STRUCTURE:")
    print("="*50)
    print("Current Cell:")
    print(f"  Obstacle: {obs['current_cell']['obstacle']}")
    print(f"  Has Items [dog, flower, notes]: {obs['current_cell']['has_item']}")
    print(f"  Is Goal: {obs['current_cell']['is_goal']}")
    print("\nNeighbors:")
    for direction in ['up', 'right', 'down', 'left']:
        print(f"  {direction}: Accessible={obs['neighbors'][direction]['accessible']}, Obstacle={obs['neighbors'][direction]['obstacle']}")
    print(f"\nGhost Relative Position: {obs['ghost_relative_pos']}")
    print("="*50)
    
    actions = [3, 2, 1, 0]
    action_names = ["up", "right", "down", "left"]
    
    for i, (action, action_name) in enumerate(zip(actions, action_names)):
        print(f"\n\nMove {i+1}: Agent moves {action_name}")
        input("Press Enter to continue...")
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        print(f"\nReward: {reward}")
        
        if terminated:
            if info.get("caught_by_ghost", False):
                print("\n🚨 GAME OVER! The ghost caught you! 🚨")
            else:
                print("\n🎉 Congratulations! You reached the goal! 🎉")
            break
    
    print("\n\nSimulation complete!")
