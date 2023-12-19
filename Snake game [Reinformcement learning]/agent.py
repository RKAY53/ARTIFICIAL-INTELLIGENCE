#Kalaipriyan R
import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        self.actions = actions
        self.Ne = Ne
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        # The initial state and action are now set to a valid starting point instead of None
        self.s = (0,) * 8  
        self.a = 0  

    def update_n(self, state, action):
        # Only update if we have a valid state and action
        if state is not None and action is not None:
            self.N[state][action] += 1

    def update_q(self, s, a, reward, s_prime):
        # Only update if we have a valid state and action
        if s is not None and a is not None:
            alpha = self.C / (self.C + self.N[s][a])
            max_q_s_prime = max(self.Q[s_prime])
            self.Q[s][a] += alpha * (reward + self.gamma * max_q_s_prime - self.Q[s][a])

    def act(self, environment, points, dead):
        current_state = self.generate_state(environment)

        # If dead, update Q-value for the last action that led to death, then reset
        if dead:
            self.update_q(self.s, self.a, -1, current_state)
            self.reset()
            return utils.RIGHT  # Return a default action

        # Choose an action based on the current state and whether we're training
        action = self.choose_action(current_state, self._train)

        # Update N-table and Q-table 
        if self._train:
            self.update_n(current_state, action)
            reward = self.calculate_reward(points)
            self.update_q(self.s, self.a, reward, current_state)

        # Update state and action for the next step
        self.s = current_state
        self.a = action

        return action

    def choose_action(self, state, is_training):
        if is_training:
            # During training, use an exploration function to determine the action
            # This is a modification to avoid using the actions.index() which is error-prone
            unexplored_actions = [a for a in self.actions if self.N[state][a] <= self.Ne]
            if unexplored_actions:
                return np.random.choice(unexplored_actions)
            else:
                return self.get_best_action(state)
        else:
            # During evaluation, always exploit the best known action
            return self.get_best_action(state)

    def action_priority(self, action):
        priority = {'RIGHT': 3, 'LEFT': 2, 'DOWN': 1, 'UP': 0}
        return priority.get(action, -1)  # Default to -1 for undefined actions

    def get_best_action(self, state):
        # Check if the state is a tuple and has the right dimensions
        if not isinstance(state, tuple) or len(state) != 8:  # Assuming there are 8 dimensions in the state
            raise ValueError("Invalid state tuple.")
        
        q_values = self.Q[state]
        max_q_value = np.max(q_values)
        best_actions = [action for action in self.actions if q_values[action] == max_q_value]
        
        best_action = sorted(best_actions, key=self.action_priority, reverse=True)[0]
        
        return best_action

    def calculate_reward(self, points):
        if points > self.points:
            self.points = points
            return 1
        else:
            return -0.1  # Assume the default penalty for no food eaten
        
    def generate_state(self, environment):
        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y = environment

        food_dir_x = 0 if food_x == snake_head_x else (1 if food_x < snake_head_x else 2)
        food_dir_y = 0 if food_y == snake_head_y else (1 if food_y < snake_head_y else 2)

        adjoining_wall_x, adjoining_wall_y = 0, 0

        if snake_head_x == 1 or (rock_x == snake_head_x - 1 and rock_y == snake_head_y):
            adjoining_wall_x = 1
        elif snake_head_x == self.display_width - 2 or (rock_x == snake_head_x + 1 and rock_y == snake_head_y):
            adjoining_wall_x = 2

        if snake_head_y == 1 or (rock_y == snake_head_y - 1 and rock_x == snake_head_x):
            adjoining_wall_y = 1
        elif snake_head_y == self.display_height - 2 or (rock_y == snake_head_y + 1 and rock_x == snake_head_x):
            adjoining_wall_y = 2

        if rock_x in [snake_head_x - 1, snake_head_x + 1] and rock_y == snake_head_y:
            adjoining_wall_x = 1
        if rock_y in [snake_head_y - 1, snake_head_y + 1] and rock_x == snake_head_x:
            adjoining_wall_y = 1

        adjoining_body_top = 1 if (snake_head_x, snake_head_y - 1) in snake_body else 0
        adjoining_body_bottom = 1 if (snake_head_x, snake_head_y + 1) in snake_body else 0
        adjoining_body_left = 1 if (snake_head_x - 1, snake_head_y) in snake_body else 0
        adjoining_body_right = 1 if (snake_head_x + 1, snake_head_y) in snake_body else 0

        state = (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y,
                adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        
        return state