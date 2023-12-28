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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
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
    from util import Stack
    # from game import Directions
    # s = Directions.SOUTH
    # n = Directions.NORTH
    # w = Directions.WEST
    # e = Directions.EAST
    if problem.isGoalState(problem.getStartState()):
        return []
    
    start = problem.getStartState()
    fringes = Stack()
    closed = set()
    fringes.push((start, [])) # (state, actions)
    
    
    while (True):
        # print("Fringe Length: "+ str(len(fringes.list)))
        if fringes.isEmpty():
            return []
        candidate = fringes.pop() # only one candidate is out..
        
        if problem.isGoalState(candidate[0]): # the candidate "state" dim
            return candidate[1] # the candidate "action dim"
        if candidate[0] not in closed:
            closed.add(candidate[0])
            for child in problem.getSuccessors(candidate[0]):
                child_state, child_action, _ = child
                if child_state not in closed:
                    fringes.push((child_state, candidate[1]+[child_action]))
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    if problem.isGoalState(problem.getStartState()):
        return []
    fringes = Queue()
    start = problem.getStartState()
    closed = set()
    fringes.push((start, [])) # (state, actions)
    
    
    while (True):
        # if len(fringes.list[0][1])<=10: 
        #     print(fringes.list)  
        # print("Fringe Length: "+ str(len(fringes.list)))
        if fringes.isEmpty():
            return []
        candidate = fringes.pop()
        if problem.isGoalState(candidate[0]):
            return candidate[1] # actions
        if candidate[0] not in closed:
            closed.add(candidate[0])
            for child in problem.getSuccessors(candidate[0]):
                child_state, child_action, _ = child
                if child_state not in closed:
                    fringes.push((child_state, candidate[1]+[child_action]))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    import heapq
    def update_modified(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i[0] == item[0]:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
    
    
    
    if problem.isGoalState(problem.getStartState()):
        return []
    fringes = PriorityQueue()
    start = problem.getStartState()
    closed = set()
    fringes.push((start, [], 0), priority=0) # (state, actions)
    
    
    while (True):
        # print("Fringe Length: "+ str(len(fringes.list)))
        if fringes.isEmpty():
            return []
        # print("Fringes: ")
        candidate = fringes.pop()
        
            
        # print("Candidate: "+str(candidate[0])+" with cost: "+str(candidate[2]))
        if problem.isGoalState(candidate[0]):
            return candidate[1]
        if candidate[0] not in closed:
            closed.add(candidate[0])
            for child in problem.getSuccessors(candidate[0]):
                child_state, child_action, child_cost = child
                if child_state not in closed:
                    fringes.update((child_state, candidate[1]+[child_action], 
                                    problem.getCostOfActions(candidate[1]+[child_action])), 
                                priority=problem.getCostOfActions(candidate[1]+[child_action]))
                    # print("Child, new: "+str(child_state)+", "+str(child_action)+", "+str(child_cost))
                    
                # elif problem.isGoalState(child[0]):
                #     update_modified(fringes, (child_state, candidate[1]+[child_action], 
                #                             problem.getCostOfActions(candidate[1]+[child_action])), 
                #                     problem.getCostOfActions(candidate[1]+[child_action]))
                    
                    # print("Goal Child, updated: "+str(child_state)+", "+str(child_action)+", "+str(child_cost))
            # print("-----------------")
                
    # util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    import heapq
    def update_modified(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i[0] == item[0]:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
    
    
    
    if problem.isGoalState(problem.getStartState()):
        return []
    fringes = PriorityQueue()
    start = problem.getStartState()
    closed = set()
    fringes.push((start, [], 0), priority=0+heuristic(start, problem)) # (state, actions)
    
    
    while (True):
        # print("Fringe Length: "+ str(len(fringes.list)))
        if fringes.isEmpty():
            return []
        # print("Fringes: ")
        candidate = fringes.pop()
        
            
        # print("Candidate: "+str(candidate[0])+" with cost: "+str(candidate[2]))
        if problem.isGoalState(candidate[0]):
            return candidate[1]
        if candidate[0] not in closed:
            closed.add(candidate[0])
            for child in problem.getSuccessors(candidate[0]):
                child_state, child_action, child_one_cost = child
                if child_state not in closed:
                    fringes.update((child_state, candidate[1]+[child_action], 
                                    problem.getCostOfActions(candidate[1]+[child_action])), 
                                    problem.getCostOfActions(candidate[1]+[child_action])+heuristic(child_state, problem))



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
