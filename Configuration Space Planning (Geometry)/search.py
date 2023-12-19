# search.py
# Kalaipriyan R
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze


# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


import heapq
import maze
from collections import deque


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI

def astar(maze):
    start = maze.get_start()
    
    explored = set()
    frontier = [] 
    heapq.heappush(frontier, start)
    visited_states = {start:(None, 0)}   

    while frontier:
        current_state = heapq.heappop(frontier)
        if current_state.is_goal():
            return backtrack(visited_states, current_state)

        for neighbor in current_state.get_neighbors():
            g = neighbor.dist_from_start 
            if neighbor not in visited_states or g < visited_states[neighbor][1]:
                visited_states[neighbor] = (current_state,g)
                heapq.heappush(frontier, neighbor)

    return None

def backtrack(visited_states, current_state):
    path = []
    while current_state is not None:
        path.append(current_state)
        current_state=visited_states[current_state][0]
    return list(reversed(path))