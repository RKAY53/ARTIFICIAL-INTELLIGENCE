#Kalaipriyan R

from abc import ABC, abstractmethod
from utils import is_english_word, levenshteinDistance
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

    def __lt__(self, other):
        if self.tiebreak_idx < other.tiebreak_idx:
            return True
        elif self.tiebreak_idx == other.tiebreak_idx:
            return (self.dist_from_start + self.h) < (other.dist_from_start + other.h)
        return False

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

class WordLadderState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic):
        super().__init__(state, goal, dist_from_start, use_heuristic)

    def get_neighbors(self):
        nbr_states = []
        for word_idx in range(len(self.state)):
            prefix = self.state[:word_idx]
            suffix = self.state[word_idx + 1:]
            for c_idx in range(97, 97 + 26):
                c = chr(c_idx)
                potential_nbr = prefix + c + suffix
                if is_english_word(potential_nbr):
                    new_state = WordLadderState(potential_nbr, self.goal, 
                                                dist_from_start=self.dist_from_start + 1, use_heuristic=self.use_heuristic)
                    nbr_states.append(new_state)
        return nbr_states

    def is_goal(self):
        return self.state == self.goal

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def compute_heuristic(self):
        return levenshteinDistance(self.state, self.goal)

    def __lt__(self, other):
        if self.tiebreak_idx < other.tiebreak_idx:
            return True
        elif self.tiebreak_idx == other.tiebreak_idx:
            return (self.dist_from_start + self.h) < (other.dist_from_start + other.h)
        return False

    def __str__(self):
        return self.state

    def __repr__(self):
        return self.state

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class EightPuzzleState(AbstractState):
    def __init__(self, state, goal, dist_from_start, use_heuristic, zero_loc):
        super().__init__(state, goal, dist_from_start, use_heuristic)
        self.zero_loc = zero_loc
    
    def get_neighbors(self):
        nbr_states = []
        zero_row, zero_col = self.zero_loc
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dr, dc in directions:
            new_row, new_col = zero_row + dr, zero_col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = [row.copy() for row in self.state]
                new_state[zero_row][zero_col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[zero_row][zero_col]
                new_zero_loc = (new_row, new_col)
                nbr_states.append(EightPuzzleState(new_state, self.goal, 
                                                   dist_from_start=self.dist_from_start + 1, 
                                                   use_heuristic=self.use_heuristic, 
                                                   zero_loc=new_zero_loc))
        return nbr_states

    def is_goal(self):
        return self.state == self.goal

    def __hash__(self):
        return hash(tuple([item for sublist in self.state for item in sublist]))

    def __eq__(self, other):
        return self.state == other.state

    def compute_heuristic(self):
        total = 0
        for i in range(3):
            for j in range(3):
                if self.state[i][j] != 0:
                    value = self.state[i][j]
                    goal_row, goal_col = divmod(value, 3)
                    total += manhattan((i, j), (goal_row, goal_col))
        return total

    def __lt__(self, other):
        if self.tiebreak_idx < other.tiebreak_idx:
            return True
        elif self.tiebreak_idx == other.tiebreak_idx:
            return (self.dist_from_start + self.h) < (other.dist_from_start + other.h)
        return False

    def __str__(self):
        return "\n---\n" + "\n".join([" ".join([str(r) for r in c]) for c in self.state])

    def __repr__(self):
        return "\n---\n" + "\n".join([" ".join([str(r) for r in c]) for c in self.state])