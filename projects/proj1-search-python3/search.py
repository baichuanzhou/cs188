# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    """
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    util.raiseNotDefined()
    """
    start_state = problem.getStartState()
    fringe = util.Stack()
    fringe.push(Node((start_state, [], 1)))
    closed = set()
    while True:
        if fringe.isEmpty():
            return []
        state = fringe.pop()
        if problem.isGoalState(state.loc):
            return state.path
        if state.loc not in closed:
            closed.add(state.loc)
            for child in problem.getSuccessors(state.loc):
                child_node = Node(child)
                if child_node.loc not in closed:
                    child_node.path = state.path + [child_node.dir]
                    fringe.push(child_node)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    start_state = problem.getStartState()
    fringe = util.Queue()
    fringe.push(Node((start_state, [], 1)))
    closed = set()
    while True:
        if fringe.isEmpty():
            return []
        state = fringe.pop()
        if problem.isGoalState(state.loc):
            return state.path
        if state.loc not in closed:
            closed.add(state.loc)
            for child in problem.getSuccessors(state.loc):
                child_node = Node(child)
                if child_node.loc not in closed:
                    child_node.path = state.path + [child_node.dir]
                    fringe.push(child_node)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    fringe = util.PriorityQueue()
    starter_state = problem.getStartState()
    starter_node = Node((starter_state, [], 0))
    fringe.update(starter_node, starter_node.ucs_heuristic)
    closed = set()
    while True:
        if fringe.isEmpty():
            return []
        state = fringe.pop()
        if problem.isGoalState(state.loc):
            return state.path
        if state.loc not in closed:
            closed.add(state.loc)
            for child in problem.getSuccessors(state.loc):
                child_node = Node(child)
                if child_node.loc not in closed:
                    child_node.path = state.path + [child_node.dir]
                    child_node.ucs_heuristic += state.ucs_heuristic
                    fringe.update(child_node, child_node.ucs_heuristic)



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    starter_state = problem.getStartState()
    fringe = util.PriorityQueue()
    starter_node = Node((starter_state, [], 0))
    fringe.update(starter_node, starter_node.astar_heuristic)
    closed = set()
    while True:
        if fringe.isEmpty():
            return []
        state = fringe.pop()
        if problem.isGoalState(state.loc):
            return state.path
        if state.loc not in closed:
            closed.add(state.loc)
            for child in problem.getSuccessors(state.loc):
                child_node = Node(child)
                if child_node.loc not in closed:
                    child_node.ucs_heuristic += state.ucs_heuristic
                    child_node.path = state.path + [child_node.dir]
                    child_node.astar_heuristic = heuristic(child_node.loc, problem)
                    fringe.push(child_node, child_node.astar_heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


class Node:
    def __init__(self, state):
        self.loc = state[0]
        self.dir = state[1]
        self.cost = state[2]
        self.path = []
        self.ucs_heuristic = self.cost
        self.astar_heuristic = 0