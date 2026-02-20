import gymnasium as gym
import numpy as np
import pygame
import sys

class CustomGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="human", slip_probability=0.2):
        super().__init__()
        self.render_mode = render_mode
        self.slip_probability = slip_probability  # Chance to move perpendicular to intended direction
        self.rows = 4
        self.cols = 5
        self.observation_space = gym.spaces.Dict({
            "current_cell": gym.spaces.Dict({
                "colour": gym.spaces.Discrete(3),  # 0=none, 1=red, 2=green
                "has_item": gym.spaces.MultiBinary(3),  # [dog, flower, notes]
                "is_goal": gym.spaces.Discrete(2),
                "text": gym.spaces.Text(max_length=10),
            }),
            "neighbors": gym.spaces.Dict({
                "up": gym.spaces.Dict({
                    "accessible": gym.spaces.Discrete(2),
                    "colour": gym.spaces.Discrete(3),
                }),
                "right": gym.spaces.Dict({
                    "accessible": gym.spaces.Discrete(2),
                    "colour": gym.spaces.Discrete(3),
                }),
                "down": gym.spaces.Dict({
                    "accessible": gym.spaces.Discrete(2),
                    "colour": gym.spaces.Discrete(3),
                }),
                "left": gym.spaces.Dict({
                    "accessible": gym.spaces.Discrete(2),
                    "colour": gym.spaces.Discrete(3),
                }),
            }),
            "ghost_relative_pos": gym.spaces.Box(low=-4, high=4, shape=(2,), dtype=np.int32),
        })
        
        self.action_space = gym.spaces.Discrete(4)  # 0: left, 1: down, 2: right, 3: up

        # Use object dtype to store complex field information
        self.grid = np.empty((self.rows, self.cols), dtype=object)
        self._setup_grid()
        self._setup_walls()
        self.agent_pos = [2, 0]
        self.start_pos = [2, 0]
        self.ghost_pos = [0, 3]
        self.ghost_start_pos = [0, 3]
        self.step_count = 0
        self.current_turn = 0  # 0 = agent's turn, 1 = ghost's turn
        
        # Pygame setup
        self.cell_size = 100
        self.wall_thickness = 6
        self.window_width = self.cols * self.cell_size
        self.window_height = self.rows * self.cell_size + 120  # Extra space for info panel
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        
        # Colors
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 100, 100),
            'red_dark': (200, 50, 50),
            'green': (100, 200, 100),
            'green_dark': (50, 150, 50),
            'yellow': (255, 220, 100),
            'blue': (100, 150, 255),
            'purple': (180, 100, 220),
            'cyan': (100, 220, 220),
            'gray': (200, 200, 200),
            'dark_gray': (80, 80, 80),
            'orange': (255, 180, 100),
        }

    def _setup_grid(self):
        # Each cell is a dictionary with: colour, items, is_goal, is_start
        # colour: 0=none, 1=red, 2=green
        # items: list of items (dog, flower, notes)
        
        self.grid[0, 0] = {'colour': 2, 'items': ['dog'], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[0, 1] = {'colour': 1, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[0, 2] = {'colour': 0, 'items': [], 'is_goal': False, 'is_start': True, 'text': 'Start'}
        self.grid[0, 3] = {'colour': 0, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[0, 4] = {'colour': 0, 'items': ['dog', 'one_note'], 'is_goal': False, 'is_start': False, 'text': ''}
        
        self.grid[1, 0] = {'colour': 0, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[1, 1] = {'colour': 2, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[1, 2] = {'colour': 0, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[1, 3] = {'colour': 0, 'items': ['flower'], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[1, 4] = {'colour': 2, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        
        self.grid[2, 0] = {'colour': 1, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[2, 1] = {'colour': 2, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[2, 2] = {'colour': 0, 'items': ['one_note'], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[2, 3] = {'colour': 1, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[2, 4] = {'colour': 0, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        
        self.grid[3, 0] = {'colour': 1, 'items': [], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[3, 1] = {'colour': 0, 'items': [], 'is_goal': True, 'is_start': False, 'text': 'Ziel'}
        self.grid[3, 2] = {'colour': 2, 'items': ['two_notes'], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[3, 3] = {'colour': 0, 'items': ['flower'], 'is_goal': False, 'is_start': False, 'text': ''}
        self.grid[3, 4] = {'colour': 0, 'items': [], 'is_goal': True, 'is_start': False, 'text': 'Ziel'}
    
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
        self.current_turn = 0  # Agent moves first
        return self._get_obs(), {"current_turn": "agent"}

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
    
    def _get_ghost_obs(self):
        """Return observation from the ghost's perspective"""
        row, col = self.ghost_pos
        current_cell = self.grid[row, col]
        
        current_obs = {
            "colour": current_cell['colour'],
            "has_item": np.array([
                1 if 'dog' in current_cell['items'] else 0,
                1 if 'flower' in current_cell['items'] else 0,
                1 if any(note in current_cell['items'] for note in ['one_note', 'two_notes']) else 0
            ], dtype=np.int8),
            "is_goal": 1 if current_cell['is_goal'] else 0,
        }
        
        neighbors_obs = {}
        for direction, (dr, dc) in [("up", (-1, 0)), ("right", (0, 1)), ("down", (1, 0)), ("left", (0, -1))]:
            neighbor_pos = [row + dr, col + dc]
            accessible = self._is_move_valid(self.ghost_pos, neighbor_pos)
            neighbor_cell = self._get_cell_info(row + dr, col + dc)
            neighbors_obs[direction] = {
                "accessible": 1 if accessible else 0,
                "colour": neighbor_cell['colour'] if neighbor_cell else 0,
            }
        
        # Agent position relative to ghost (opposite of ghost_relative_pos)
        agent_relative = np.array([
            self.agent_pos[0] - self.ghost_pos[0],
            self.agent_pos[1] - self.ghost_pos[1]
        ], dtype=np.int32)
        
        return {
            "current_cell": current_obs,
            "neighbors": neighbors_obs,
            "agent_relative_pos": agent_relative,
        }
    
    def _move_ghost_internal(self):
        """Move the ghost using built-in chase strategy (used when no ghost agent provided)"""
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
    
    def move_ghost(self, ghost_action):
        """
        Move the ghost with an externally provided action.
        
        Args:
            ghost_action (int): Action for ghost (0=left, 1=down, 2=right, 3=up)
        """
        self.ghost_pos = self._move_entity(self.ghost_pos, ghost_action)
    
    def get_reward_structure(self):
        """
        Get the reward structure for this environment.
        
        Returns:
            dict: Dictionary describing all rewards and their values
        """
        return {
            "step_penalty": -1,
            "caught_by_ghost": -50,
            "reached_goal": 100,
            "slip_probability": self.slip_probability,
            "terminal_states": ["caught_by_ghost", "reached_goal"]
        }
    
    def calculate_reward(self, caught_by_ghost=False):
        """
        Calculate reward based on current game state.
        
        Reward Structure:
        ----------------
        - Caught by ghost: -50 (terminal)
        - Reached goal: +100 (terminal)
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
        
        return reward, terminated, info
    
    def _apply_slip(self, intended_action):
        """
        Apply slip probability to potentially change the action to a perpendicular direction.
        
        Args:
            intended_action (int): The action the agent intended to take (0=left, 1=down, 2=right, 3=up)
            
        Returns:
            tuple: (actual_action, slipped) - The actual action taken and whether a slip occurred
        """
        if self.slip_probability <= 0:
            return intended_action, False
        
        # Perpendicular actions for each direction
        # left(0) and right(2) are perpendicular to up(3) and down(1)
        perpendicular = {
            0: [1, 3],  # left -> can slip to down or up
            1: [0, 2],  # down -> can slip to left or right
            2: [1, 3],  # right -> can slip to down or up
            3: [0, 2],  # up -> can slip to left or right
        }
        
        if self.np_random.random() < self.slip_probability:
            # Slip! Choose one of the perpendicular directions randomly
            actual_action = self.np_random.choice(perpendicular[intended_action])
            return actual_action, True
        
        return intended_action, False
    
    def step(self, action):
        """
        Execute one step in the environment (alternating turns).
        
        Args:
            action (int): Action for current entity (0=left, 1=down, 2=right, 3=up)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        info = {}
        reward = 0
        terminated = False
        action_names = {0: "left", 1: "down", 2: "right", 3: "up"}
        
        if self.current_turn == 0:
            # Agent's turn
            self.step_count += 1
            
            # Apply slip probability - agent might move perpendicular to intended direction
            actual_action, slipped = self._apply_slip(action)
            
            # Move the agent with the (potentially modified) action
            self.agent_pos = self._move_entity(self.agent_pos, actual_action)
            
            # Add slip info
            if slipped:
                info['slipped'] = True
                info['intended_action'] = action_names[action]
                info['actual_action'] = action_names[actual_action]
            
            # Check if agent reached goal (only check on agent's turn)
            current_cell = self.grid[self.agent_pos[0], self.agent_pos[1]]
            if self.agent_pos == self.ghost_pos:
                reward = self.get_reward_structure()["caught_by_ghost"]
                terminated = True
                info['caught_by_ghost'] = True
            elif current_cell['is_goal']:
                reward = self.get_reward_structure()["reached_goal"]
                terminated = True
                info['reached_goal'] = True
            else:
                # Apply step penalty
                reward_structure = self.get_reward_structure()
                reward = reward_structure["step_penalty"]
            
            # Switch to ghost's turn
            self.current_turn = 1
            info['current_turn'] = 'ghost'
            info['mover'] = 'agent'
            
        else:
            # Ghost's turn
            self.move_ghost(action)
            
            # Check if ghost caught the agent
            if self.agent_pos == self.ghost_pos:
                reward = self.get_reward_structure()["caught_by_ghost"]
                terminated = True
                info['caught_by_ghost'] = True
            
            # Switch to agent's turn
            self.current_turn = 0
            info['current_turn'] = 'agent'
            info['mover'] = 'ghost'
        
        return self._get_obs(), reward, terminated, False, info
    
    def get_current_turn(self):
        """Return whose turn it is: 'agent' or 'ghost'"""
        return 'agent' if self.current_turn == 0 else 'ghost'

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
            "colour": current_cell['colour'],
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
            "colour": up_cell['colour'] if up_cell else 0,
        }
        
        right_pos = [row, col + 1]
        right_accessible = self._is_move_valid(self.agent_pos, right_pos)
        right_cell = self._get_cell_info(row, col + 1)
        neighbors_obs["right"] = {
            "accessible": 1 if right_accessible else 0,
            "colour": right_cell['colour'] if right_cell else 0,
        }
        
        down_pos = [row + 1, col]
        down_accessible = self._is_move_valid(self.agent_pos, down_pos)
        down_cell = self._get_cell_info(row + 1, col)
        neighbors_obs["down"] = {
            "accessible": 1 if down_accessible else 0,
            "colour": down_cell['colour'] if down_cell else 0,
        }
        
        left_pos = [row, col - 1]
        left_accessible = self._is_move_valid(self.agent_pos, left_pos)
        left_cell = self._get_cell_info(row, col - 1)
        neighbors_obs["left"] = {
            "accessible": 1 if left_accessible else 0,
            "colour": left_cell['colour'] if left_cell else 0,
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

    def _init_pygame(self):
        """Initialize Pygame if not already initialized"""
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Custom Grid Environment")
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
    
    def _draw_crosshatch(self, surface, rect, color, line_spacing=8):
        """Draw a crosshatch pattern inside a rectangle"""
        x, y, w, h = rect
        # Draw diagonal lines (top-left to bottom-right)
        for i in range(-h, w, line_spacing):
            start_x = max(x, x + i)
            start_y = max(y, y - i)
            end_x = min(x + w, x + i + h)
            end_y = min(y + h, y - i + w)
            if start_x < end_x:
                pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), 2)
        
        # Draw diagonal lines (top-right to bottom-left)
        for i in range(0, w + h, line_spacing):
            start_x = min(x + w, x + i)
            start_y = max(y, y + i - w)
            end_x = max(x, x + i - h)
            end_y = min(y + h, y + i)
            if start_x > end_x:
                pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), 2)
    
    def _draw_cell(self, row, col):
        """Draw a single cell with its contents"""
        x = col * self.cell_size
        y = row * self.cell_size
        cell = self.grid[row, col]
        margin = 4
        
        # Draw base cell (white background)
        pygame.draw.rect(self.screen, self.colors['white'], 
                        (x + margin, y + margin, self.cell_size - 2*margin, self.cell_size - 2*margin))
        
        # Draw colour pattern if present
        if cell['colour'] == 1:  # Red colour
            self._draw_crosshatch(self.screen, 
                                 (x + margin, y + margin, self.cell_size - 2*margin, self.cell_size - 2*margin),
                                 self.colors['red'])
        elif cell['colour'] == 2:  # Green colour
            self._draw_crosshatch(self.screen, 
                                 (x + margin, y + margin, self.cell_size - 2*margin, self.cell_size - 2*margin),
                                 self.colors['green'])
        
        # Draw text
        if cell['text']:
            text = self.font.render(cell['text'], True, self.colors['black'])
            text_rect = text.get_rect(center=(x + self.cell_size // 2, y + self.cell_size // 2))
            self.screen.blit(text, text_rect)
        
        # Draw items
        if cell['items'] and not cell['is_goal'] and not cell['is_start']:
            item_y_offset = 0
            note_offset = 0  # Separate offset for notes in top right
            for item in cell['items']:
                if 'dog' in item:
                    self._draw_dog(x + self.cell_size // 2, y + self.cell_size // 2 + item_y_offset)
                    item_y_offset += 20
                elif 'flower' in item:
                    self._draw_flower(x + self.cell_size // 2, y + self.cell_size // 2 + item_y_offset)
                    item_y_offset += 20
                elif 'one_note' in item:
                    # Position in top right corner
                    self._draw_note(x + self.cell_size - 20, y + 20 + note_offset, single=True)
                    note_offset += 25
                elif 'two_notes' in item:
                    # Position in top right corner
                    self._draw_note(x + self.cell_size - 25, y + 20 + note_offset, single=False)
                    note_offset += 25
    
    def _draw_dog(self, cx, cy):
        """Draw a simple dog icon"""
        # Body
        pygame.draw.ellipse(self.screen, self.colors['dark_gray'], (cx - 20, cy - 10, 40, 25))
        # Head
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (cx - 15, cy - 15), 12)
        # Ears
        pygame.draw.ellipse(self.screen, self.colors['dark_gray'], (cx - 28, cy - 25, 10, 15))
        pygame.draw.ellipse(self.screen, self.colors['dark_gray'], (cx - 12, cy - 25, 10, 15))
        # Eyes
        pygame.draw.circle(self.screen, self.colors['white'], (cx - 18, cy - 17), 3)
        pygame.draw.circle(self.screen, self.colors['white'], (cx - 12, cy - 17), 3)
        # Tail
        pygame.draw.arc(self.screen, self.colors['dark_gray'], (cx + 10, cy - 20, 20, 25), 0, 2, 3)
    
    def _draw_flower(self, cx, cy):
        """Draw a simple flower icon"""
        # Petals
        petal_color = self.colors['white']
        for angle in range(0, 360, 60):
            rad = np.radians(angle)
            px = cx + int(15 * np.cos(rad))
            py = cy + int(15 * np.sin(rad))
            pygame.draw.circle(self.screen, petal_color, (px, py), 10)
            pygame.draw.circle(self.screen, self.colors['dark_gray'], (px, py), 10, 1)
        # Center
        pygame.draw.circle(self.screen, self.colors['yellow'], (cx, cy), 8)
        pygame.draw.circle(self.screen, self.colors['orange'], (cx, cy), 8, 2)
    
    def _draw_note(self, cx, cy, single=True):
        """Draw musical note(s)"""
        if single:
            # Single note
            pygame.draw.ellipse(self.screen, self.colors['black'], (cx - 8, cy, 12, 10))
            pygame.draw.line(self.screen, self.colors['black'], (cx + 3, cy + 5), (cx + 3, cy - 25), 3)
            pygame.draw.arc(self.screen, self.colors['black'], (cx, cy - 30, 15, 15), 3.5, 6, 3)
        else:
            # Double notes
            pygame.draw.ellipse(self.screen, self.colors['black'], (cx - 18, cy, 12, 10))
            pygame.draw.ellipse(self.screen, self.colors['black'], (cx + 2, cy, 12, 10))
            pygame.draw.line(self.screen, self.colors['black'], (cx - 7, cy + 5), (cx - 7, cy - 20), 3)
            pygame.draw.line(self.screen, self.colors['black'], (cx + 13, cy + 5), (cx + 13, cy - 20), 3)
            pygame.draw.line(self.screen, self.colors['black'], (cx - 7, cy - 20), (cx + 13, cy - 20), 3)
    
    def _draw_agent(self, row, col):
        """Draw the agent (robot with GPS)"""
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        
        # Robot body
        pygame.draw.rect(self.screen, self.colors['gray'], (x - 25, y - 20, 50, 45), border_radius=5)
        pygame.draw.rect(self.screen, self.colors['dark_gray'], (x - 25, y - 20, 50, 45), 2, border_radius=5)
        
        # Antenna
        pygame.draw.line(self.screen, self.colors['dark_gray'], (x, y - 20), (x, y - 35), 3)
        pygame.draw.circle(self.screen, self.colors['red'], (x, y - 35), 5)
        
        # Eyes (screen)
        pygame.draw.rect(self.screen, self.colors['cyan'], (x - 18, y - 12, 36, 20), border_radius=3)
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (x - 8, y - 2), 5)
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (x + 8, y - 2), 5)
        
        # GPS text
        gps_text = self.small_font.render("GPS", True, self.colors['dark_gray'])
        gps_rect = gps_text.get_rect(center=(x, y + 15))
        self.screen.blit(gps_text, gps_rect)
        
        # Wheels
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (x - 18, y + 28), 8)
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (x + 18, y + 28), 8)
    
    def _draw_ghost(self, row, col):
        """Draw the ghost (Pac-Man style)"""
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        
        # Ghost body (Pac-Man style)
        color = self.colors['cyan']
        
        # Main body (rounded top)
        pygame.draw.circle(self.screen, color, (x, y - 5), 25)
        pygame.draw.rect(self.screen, color, (x - 25, y - 5, 50, 30))
        
        # Wavy bottom
        for i in range(5):
            wave_x = x - 20 + i * 10
            pygame.draw.circle(self.screen, color, (wave_x, y + 25), 5)
        
        # Eyes (white with dark pupils)
        pygame.draw.circle(self.screen, self.colors['white'], (x - 10, y - 8), 8)
        pygame.draw.circle(self.screen, self.colors['white'], (x + 10, y - 8), 8)
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (x - 8, y - 6), 4)
        pygame.draw.circle(self.screen, self.colors['dark_gray'], (x + 12, y - 6), 4)
    
    def _draw_walls(self):
        """Draw all walls"""
        wall_color = self.colors['black']
        
        # Draw vertical walls (between columns)
        for row in range(self.rows):
            for col in range(self.cols - 1):
                if self.walls_vertical[row, col]:
                    x = (col + 1) * self.cell_size
                    y = row * self.cell_size
                    pygame.draw.rect(self.screen, wall_color, 
                                   (x - self.wall_thickness // 2, y, self.wall_thickness, self.cell_size))
        
        # Draw horizontal walls (between rows)
        for row in range(self.rows - 1):
            for col in range(self.cols):
                if self.walls_horizontal[row, col]:
                    x = col * self.cell_size
                    y = (row + 1) * self.cell_size
                    pygame.draw.rect(self.screen, wall_color, 
                                   (x, y - self.wall_thickness // 2, self.cell_size, self.wall_thickness))
    
    def _draw_grid_lines(self):
        """Draw the grid lines"""
        line_color = self.colors['gray']
        
        # Vertical lines
        for col in range(self.cols + 1):
            x = col * self.cell_size
            pygame.draw.line(self.screen, line_color, (x, 0), (x, self.rows * self.cell_size), 1)
        
        # Horizontal lines
        for row in range(self.rows + 1):
            y = row * self.cell_size
            pygame.draw.line(self.screen, line_color, (0, y), (self.window_width, y), 1)
    
    def _draw_info_panel(self):
        """Draw the information panel at the bottom"""
        panel_y = self.rows * self.cell_size
        pygame.draw.rect(self.screen, self.colors['dark_gray'], (0, panel_y, self.window_width, 120))
        
        # Step counter
        step_text = self.font.render(f"Step: {self.step_count}", True, self.colors['white'])
        self.screen.blit(step_text, (10, panel_y + 10))
        
        # Current position
        pos_text = self.small_font.render(f"Agent: ({self.agent_pos[0]}, {self.agent_pos[1]})", True, self.colors['white'])
        self.screen.blit(pos_text, (10, panel_y + 45))
        
        # Ghost position
        ghost_text = self.small_font.render(f"Ghost: ({self.ghost_pos[0]}, {self.ghost_pos[1]})", True, self.colors['cyan'])
        self.screen.blit(ghost_text, (10, panel_y + 70))
        
        # Distance
        distance = abs(self.agent_pos[0] - self.ghost_pos[0]) + abs(self.agent_pos[1] - self.ghost_pos[1])
        dist_text = self.small_font.render(f"Distance: {distance}", True, self.colors['yellow'])
        self.screen.blit(dist_text, (10, panel_y + 95))
        
        # Turn indicator
        turn_name = "Agent's Turn" if self.current_turn == 0 else "Ghost's Turn"
        turn_color = self.colors['yellow'] if self.current_turn == 0 else self.colors['cyan']
        turn_text = self.font.render(turn_name, True, turn_color)
        self.screen.blit(turn_text, (200, panel_y + 10))
        
        # Current cell info
        current_cell = self.grid[self.agent_pos[0], self.agent_pos[1]]
        colour_name = 'None' if current_cell['colour'] == 0 else 'Red' if current_cell['colour'] == 1 else 'Green'
        cell_text = self.small_font.render(f"Cell colour: {colour_name}", True, self.colors['white'])
        self.screen.blit(cell_text, (200, panel_y + 45))
        
        items_str = ', '.join(current_cell['items']) if current_cell['items'] else 'None'
        items_text = self.small_font.render(f"Items: {items_str}", True, self.colors['white'])
        self.screen.blit(items_text, (200, panel_y + 70))
        
        goal_text = self.small_font.render(f"Goal: {'Yes' if current_cell['is_goal'] else 'No'}", True, 
                                          self.colors['yellow'] if current_cell['is_goal'] else self.colors['white'])
        self.screen.blit(goal_text, (200, panel_y + 95))
    
    def render(self):
        """Render the environment using Pygame"""
        self._init_pygame()
        
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
        
        # Clear screen
        self.screen.fill(self.colors['white'])
        
        # Draw grid lines first
        self._draw_grid_lines()
        
        # Draw all cells
        for row in range(self.rows):
            for col in range(self.cols):
                self._draw_cell(row, col)
        
        # Draw walls
        self._draw_walls()
        
        # Draw ghost (if not same position as agent)
        if self.agent_pos != self.ghost_pos:
            self._draw_ghost(self.ghost_pos[0], self.ghost_pos[1])
        
        # Draw agent
        self._draw_agent(self.agent_pos[0], self.agent_pos[1])
        
        # If both on same cell, draw collision indicator
        if self.agent_pos == self.ghost_pos:
            x = self.agent_pos[1] * self.cell_size + self.cell_size // 2
            y = self.agent_pos[0] * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, self.colors['red'], (x, y), 45, 5)
            bang_text = self.font.render("!", True, self.colors['red'])
            self.screen.blit(bang_text, (x - 5, y - 15))
        
        # Draw info panel
        self._draw_info_panel()
        
        # Draw border
        pygame.draw.rect(self.screen, self.colors['black'], 
                        (0, 0, self.window_width, self.rows * self.cell_size), 3)
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Clean up Pygame resources"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

gym.envs.registration.register(
    id="CustomGrid-v0",
    entry_point="Test:CustomGridEnv",
)


class ChaseGhostAgent:
    """
    Ghost agent that chases the player using simple pathfinding.
    This replicates the original built-in ghost behavior.
    """
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation):
        """
        Choose action to move toward the agent.
        
        Args:
            observation (dict): Ghost's observation with 'agent_relative_pos'
            
        Returns:
            int: Action (0=left, 1=down, 2=right, 3=up)
        """
        agent_relative = observation['agent_relative_pos']
        row_diff = agent_relative[0]  # Positive = agent is below ghost
        col_diff = agent_relative[1]  # Positive = agent is to the right of ghost
        
        neighbors = observation['neighbors']
        
        # Prioritize direction with larger distance
        if abs(row_diff) >= abs(col_diff):
            # Try vertical first
            if row_diff > 0 and neighbors['down']['accessible']:
                return 1  # down
            elif row_diff < 0 and neighbors['up']['accessible']:
                return 3  # up
            # Fall back to horizontal
            if col_diff > 0 and neighbors['right']['accessible']:
                return 2  # right
            elif col_diff < 0 and neighbors['left']['accessible']:
                return 0  # left
        else:
            # Try horizontal first
            if col_diff > 0 and neighbors['right']['accessible']:
                return 2  # right
            elif col_diff < 0 and neighbors['left']['accessible']:
                return 0  # left
            # Fall back to vertical
            if row_diff > 0 and neighbors['down']['accessible']:
                return 1  # down
            elif row_diff < 0 and neighbors['up']['accessible']:
                return 3  # up
        
        # If no good move, try any accessible direction
        for action, direction in [(3, 'up'), (2, 'right'), (1, 'down'), (0, 'left')]:
            if neighbors[direction]['accessible']:
                return action
        
        # No move possible, stay in place (return any action, won't move)
        return 0


class RandomGhostAgent:
    """Ghost agent that moves randomly."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation):
        """Return a random action."""
        return self.action_space.sample()


class AgentInterface:
    """
    Interface for AI agents to interact with the CustomGridEnv.
    The ghost is controlled internally by an AI agent (not exposed to user).
    
    Usage:
        interface = AgentInterface(render=True, slip_probability=0.2)
        
        # Reset and get initial observation
        obs = interface.reset()
        
        # Run episodes
        while not interface.is_terminated():
            action = your_agent.get_action(obs)
            obs, reward, done, info = interface.step(action)
        
        # Get episode results
        results = interface.get_episode_stats()
        interface.close()
    """
    
    def __init__(self, render=True, step_delay=100, slip_probability=0.2, ghost_agent_class=None):
        """
        Initialize the agent interface.
        
        Args:
            render (bool): Whether to render the graphical display
            step_delay (int): Milliseconds to wait between steps when rendering
            slip_probability (float): Probability of slipping to a perpendicular direction (0.0 to 1.0)
            ghost_agent_class: Class for ghost agent. If None, uses ChaseGhostAgent (internal).
        """
        self.env = CustomGridEnv(slip_probability=slip_probability)
        self.render_enabled = render
        self.step_delay = step_delay
        self.total_reward = 0
        self.terminated = False
        self.truncated = False
        self.episode_steps = 0
        self.last_info = {}
        
        # Initialize ghost agent (internal)
        if ghost_agent_class is None:
            self._ghost_agent = ChaseGhostAgent(self.env.action_space)
        else:
            self._ghost_agent = ghost_agent_class(self.env.action_space)
    
    def reset(self, seed=None):
        """
        Reset the environment for a new episode.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            dict: Initial observation for the agent
        """
        obs, info = self.env.reset(seed=seed)
        self.total_reward = 0
        self.terminated = False
        self.truncated = False
        self.episode_steps = 0
        self.last_info = info
        
        if self.render_enabled:
            self.env.render()
        
        return obs
    
    def step(self, action):
        """
        Take a step in the environment (agent moves, then ghost moves automatically).
        
        Args:
            action (int): Agent action (0=left, 1=down, 2=right, 3=up)
            
        Returns:
            tuple: (observation, reward, done, info) - standard Gym-like interface
        """
        if self.terminated or self.truncated:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")
        
        combined_info = {}
        total_step_reward = 0
        
        # Agent's turn
        obs, reward, self.terminated, self.truncated, info = self.env.step(action)
        total_step_reward += reward
        combined_info.update(info)
        
        if self.render_enabled:
            self.env.render()
            pygame.time.wait(self.step_delay)
        
        # Check if game ended on agent's turn (reached goal)
        if self.terminated:
            self.total_reward += total_step_reward
            self.episode_steps += 1
            self.last_info = combined_info
            return obs, total_step_reward, True, combined_info
        
        # Ghost's turn (internal)
        ghost_obs = self.env._get_ghost_obs()
        ghost_action = self._ghost_agent.get_action(ghost_obs)
        
        obs, reward, self.terminated, self.truncated, info = self.env.step(ghost_action)
        combined_info.update(info)
        
        if self.render_enabled:
            self.env.render()
            pygame.time.wait(self.step_delay)
        
        # Add ghost catch penalty to agent reward
        if info.get('caught_by_ghost'):
            total_step_reward += reward
        
        self.total_reward += total_step_reward
        self.episode_steps += 1
        self.last_info = combined_info
        
        return obs, total_step_reward, self.terminated or self.truncated, combined_info
    
    def is_terminated(self):
        """Check if the current episode has ended."""
        return self.terminated or self.truncated
    
    def get_episode_stats(self):
        """
        Get statistics for the current/last episode.
        
        Returns:
            dict: Episode statistics
        """
        return {
            "total_reward": self.total_reward,
            "steps": self.episode_steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "reached_goal": self.last_info.get("reached_goal", False),
            "caught_by_ghost": self.last_info.get("caught_by_ghost", False),
        }
    
    def get_action_space(self):
        """Get the action space (for agent initialization)."""
        return self.env.action_space
    
    def get_observation_space(self):
        """Get the observation space (for agent initialization)."""
        return self.env.observation_space
    
    def get_reward_structure(self):
        """Get the reward structure for the environment."""
        return self.env.get_reward_structure()
    
    def close(self):
        """Clean up resources."""
        self.env.close()


class RandomAgent:
    """Example random agent for demonstration."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation):
        """Return a random action."""
        return self.action_space.sample()


def run_with_agent(agent_class=None, ghost_agent_class=None, num_episodes=1, render=True, step_delay=100, slip_probability=0.2):
    """
    Run the environment with an AI agent. Ghost is controlled internally.
    
    Args:
        agent_class: Agent class with get_action(obs) method. If None, uses RandomAgent.
        ghost_agent_class: Ghost agent class (internal). If None, uses ChaseGhostAgent.
        num_episodes (int): Number of episodes to run
        render (bool): Whether to render the display
        step_delay (int): Delay between steps in ms
        slip_probability (float): Probability of slipping to a perpendicular direction (0.0 to 1.0)
        
    Returns:
        list: List of episode statistics
    """
    interface = AgentInterface(
        render=render, 
        step_delay=step_delay, 
        slip_probability=slip_probability,
        ghost_agent_class=ghost_agent_class
    )
    
    # Initialize agent
    if agent_class is None:
        agent = RandomAgent(interface.get_action_space())
    else:
        agent = agent_class(interface.get_action_space())
    
    ghost_name = ghost_agent_class.__name__ if ghost_agent_class else "ChaseGhostAgent"
    agent_name = agent_class.__name__ if agent_class else "RandomAgent"
    
    print("\n" + "="*50)
    print("ENVIRONMENT INFO")
    print("="*50)
    print(f"Agent: {agent_name}")
    print(f"Ghost (internal): {ghost_name}")
    print(f"Mode: Alternating turns (Agent moves, then Ghost moves)")
    print(f"Action Space: {interface.get_action_space()}")
    print("Actions: 0=left, 1=down, 2=right, 3=up")
    print("\nReward Structure:")
    for key, value in interface.get_reward_structure().items():
        if key != "terminal_states":
            print(f"  {key}: {value}")
    print("="*50)
    
    all_stats = []
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        obs = interface.reset()
        
        while not interface.is_terminated():
            action = agent.get_action(obs)
            obs, reward, done, info = interface.step(action)
        
        stats = interface.get_episode_stats()
        all_stats.append(stats)
        
        result = "GOAL!" if stats["reached_goal"] else "CAUGHT!" if stats["caught_by_ghost"] else "ENDED"
        print(f"Result: {result} | Steps: {stats['steps']} | Reward: {stats['total_reward']}")
    
    interface.close()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    avg_reward = sum(s["total_reward"] for s in all_stats) / len(all_stats)
    avg_steps = sum(s["steps"] for s in all_stats) / len(all_stats)
    goals = sum(1 for s in all_stats if s["reached_goal"])
    caught = sum(1 for s in all_stats if s["caught_by_ghost"])
    print(f"Episodes: {num_episodes}")
    print(f"Goals Reached: {goals} ({100*goals/num_episodes:.1f}%)")
    print(f"Caught by Ghost: {caught} ({100*caught/num_episodes:.1f}%)")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print("="*50)
    
    return all_stats


if __name__ == "__main__":
    # Example: Run with random agent
    # The ghost is controlled internally by ChaseGhostAgent
    # 
    # To use your own agent, create a class with get_action(observation) method:
    #   class MyAgent:
    #       def __init__(self, action_space):
    #           self.action_space = action_space
    #       def get_action(self, obs):
    #           return 0  # your logic here
    
    run_with_agent(
        agent_class=None,  # Uses RandomAgent by default
        ghost_agent_class=None,  # Uses ChaseGhostAgent internally
        num_episodes=3,
        render=True,
        step_delay=200
    )
