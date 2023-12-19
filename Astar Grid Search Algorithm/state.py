#Kalaipriyan R

from utils import compute_mst_cost
from abc import ABC, abstractmethod
from itertools import count

global_index = count()

class AbstractState(ABC):
    def __init__(self, state, goal, dist_from_start=0, use_heuristic=True):
        self.state = state
        self.goal = goal
        self.tiebreak_idx = next(global_index)
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.h = self.compute_heuristic()
        else:
            self.h = 0

    @abstractmethod
    def get_neighbors(self):
        pass
    
    @abstractmethod
    def is_goal(self):
        pass
    
    @abstractmethod
    def compute_heuristic(self):
        pass
    
    @abstractmethod
    def __lt__(self, other):
        if self.tiebreak_idx < other.tiebreak_idx:
            return True

    @abstractmethod
    def __hash__(self):
        pass
    
    @abstractmethod
    def __eq__(self, other):
        pass

class SingleGoalGridState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic, maze_neighbors):
        self.maze_neighbors = maze_neighbors
        super().__init__(state, goal, dist_from_start, use_heuristic)
        
    def get_neighbors(self):
        nbr_states = []
        neighboring_grid_locs = self.maze_neighbors(*self.state)
        for loc in neighboring_grid_locs:
            nbr_states.append(SingleGoalGridState(loc, self.goal, self.dist_from_start + 1, self.use_heuristic, self.maze_neighbors))
        return nbr_states

    def is_goal(self):
        return self.state == self.goal[0]
    
    def __hash__(self):
        return hash(self.state)
    
    def __eq__(self, other):
        return self.state == other.state
    
    def compute_heuristic(self):
        return abs(self.state[0] - self.goal[0][0]) + abs(self.state[1] - self.goal[0][1])
    
    def __lt__(self, other):
        if (self.dist_from_start + self.h) < (other.dist_from_start + other.h):
            return True
        elif (self.dist_from_start + self.h) == (other.dist_from_start + other.h):
            return self.tiebreak_idx < other.tiebreak_idx
        else:
            return False
    
    def __str__(self):
        return str(self.state) + ", goal=" + str(self.goal)

    def __repr__(self):
        return str(self.state) + ", goal=" + str(self.goal)


class Maze:
    def __init__(self):
        # Define your Maze class here if it's not already defined
        pass
    
class GridState(AbstractState):
    def __init__(self, state, goals, dist_from_start, use_heuristic, maze_neighbors, mst_cache=None):
        self.goals = goals
        self.maze_neighbors = maze_neighbors
        self.mst_cache = mst_cache if mst_cache is not None else {}
        super().__init__(state, goals, dist_from_start, use_heuristic)

    def get_neighbors(self):
        nbr_states = []
        neighboring_locs = self.maze_neighbors(*self.state)
        for loc in neighboring_locs:
            if loc in self.goals:
                modified_goal = list(self.goals)
                modified_goal.remove(loc)
                
                modified_goal = tuple(modified_goal)
                nbr_states.append(GridState(loc, modified_goal, self.dist_from_start + 1, self.use_heuristic, self.maze_neighbors, self.mst_cache))
            else:
                nbr_states.append(GridState(loc, self.goals, self.dist_from_start + 1, self.use_heuristic, self.maze_neighbors, self.mst_cache))
        return nbr_states

    def is_goal(self):
        return self.state in self.goals

    def __eq__(self, other):
        return self.state == other.state and set(self.goals) == set(other.goals)

    def __hash__(self):
        return hash((self.state, tuple(self.goals)))

    def compute_heuristic(self):
        if not self.goals:
            return 0 

        if len(self.goals) == 1:
            return abs(self.state[0] - self.goals[0][0]) + abs(self.state[1] - self.goals[0][1])
        else:
            goal_set = frozenset(self.goals) 
            if goal_set not in self.mst_cache:
                self.mst_cache[goal_set] = compute_mst_cost(goal_set, lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]))
            return abs(self.state[0] - self.goals[0][0]) + abs(self.state[1] - self.goals[0][1]) + self.mst_cache[goal_set]


    def __lt__(self, other):
        if (self.dist_from_start + self.h) < (other.dist_from_start + other.h):
            return True
        elif (self.dist_from_start + self.h) == (other.dist_from_start + other.h):
            return self.tiebreak_idx < other.tiebreak_idx

    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goals)

    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goals)
    
@staticmethod
def manhattan(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1]-point2[1])