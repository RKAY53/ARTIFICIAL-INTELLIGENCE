# Kalaipriyan R
#state.py
import copy
import math
from itertools import count

# NOTE: using this global index means that if we solve multiple
#       searches consecutively the index doesn't reset to 0... this is fine
global_index = count()


# TODO VI
# Euclidean distance between two state tuples, of the form (x,y, shape)
def euclidean_distance(a, b):
    x1, y1, shape1 = a
    x2, y2, shape2 = b
    if shape1!= shape2: return 10
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def euclidean_distance2(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

from abc import ABC, abstractmethod


class AbstractState(ABC):
    def __init__(self, state, goal, dist_from_start=0., use_heuristic=True):
        self.state = state
        self.goal = goal
        # we tiebreak based on the order that the state was created/found
        self.tiebreak_idx = next(global_index)
        # dist_from_start is classically called "g" when describing A*, i.e., f = g + h
        self.dist_from_start = dist_from_start
        self.use_heuristic = use_heuristic
        if use_heuristic:
            self.h = self.compute_heuristic()
        else:
            self.h = 0
            
    @abstractmethod
    def get_neighbors(self):
        pass

    # Return True if the state is the goal
    @abstractmethod
    def is_goal(self):
        pass

    @abstractmethod
    def compute_heuristic(self):
        pass

    @abstractmethod
    def __lt__(self, other):
        # NOTE: if the two states (self and other) have the same f value, tiebreak using tiebreak_idx as below
        if self.tiebreak_idx < other.tiebreak_idx:
            return True

    @abstractmethod
    def __hash__(self):
        pass

    # __eq__ gets called during hashing collisions, without it Python checks object equality
    @abstractmethod
    def __eq__(self, other):
        pass

class MazeState(AbstractState):
    def __init__(self, state, goal, dist_from_start, maze, use_heuristic=True):
        self.maze = maze
        self.maze_neighbors = maze.get_neighbors
        super().__init__(state, goal, dist_from_start, use_heuristic)

    # TODO VI
    def get_neighbors(self):
        nbr_states = []

        neighbors = self.maze_neighbors( *self.state )

        for neighbor in neighbors:
            dist_cost = euclidean_distance( self.state, neighbor )

            neighbor_state = MazeState( state = neighbor,
                                        goal = self.goal, 
                                        dist_from_start = self.dist_from_start + dist_cost,
                                        maze = self.maze,
                                        use_heuristic = self.use_heuristic )
        
            nbr_states.append( neighbor_state )
        return nbr_states

    # TODO VI
    def is_goal(self):
        return (self.state[0], self.state[1]) in self.goal
            
    # TODO VI
    def __hash__(self):
        return hash((self.state, self.goal))

    # TODO VI
    def __eq__(self, other):
        return self.state == other.state and self.goal == other.goal

    # TODO VI
    def compute_heuristic(self):
        shortest_distance = float("infinity")
        for goal in self.goal:
            shortest_distance = min(shortest_distance, euclidean_distance(self.state, (goal[0], goal[1], self.state[2])))
        return shortest_distance

    # This method allows the heap to sort States according to f = g + h value
    # TODO VI
    def __lt__(self, other):
        if (self.dist_from_start+self.h) < (other.dist_from_start + other.h):
            return True
        elif (self.dist_from_start+self.h) > (other.dist_from_start + other.h):
            return False
        return True if self.tiebreak_idx < other.tiebreak_idx else False

    # str and repr just make output more readable when your print out states
    def __str__(self):
        return str(self.state) + ", goals=" + str(self.goal)

    def __repr__(self):
        return str(self.state) + ", goals=" + str(self.goal)
