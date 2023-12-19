#Kalaipriyan R

import heapq
from queue import PriorityQueue

def best_first_search(starting_state):
    visited_states = {starting_state: (None, 0)}
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    while frontier:
        current_state = heapq.heappop(frontier)
        
        if current_state.is_goal() or len(current_state.goal) == 0:
            return backtrack(visited_states, current_state)

        neighbors = current_state.get_neighbors()

        for neighbor_state in neighbors:
            new_dist = current_state.dist_from_start + 1
            if (neighbor_state not in visited_states) or (new_dist < visited_states[neighbor_state][1]):
                visited_states[neighbor_state] = (current_state, new_dist)
                heapq.heappush(frontier, neighbor_state)
    
    return []

def backtrack(visited_states, goal_state):
    path = []
    current_state = goal_state

    while current_state is not None:
        path.insert(0, current_state)
        current_state = visited_states[current_state][0]

    return path
