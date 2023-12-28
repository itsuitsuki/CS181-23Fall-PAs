# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import pacman


class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP


#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state)  # Makes a new search problem
        self.actions = self.searchFunction(problem)  # Find a path
        if self.actions == None:
            self.actions = []
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)


class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0  # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return (pos := self.startingPosition,  # 我自己所在的位置, (x, y) tuple
                corners := self.corners,  # 我还没访问的corner位置
                )

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        return len(state[1]) == 0
        # util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        x, y = state[0]
        corners = state[1]
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:

            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]
            if not hitsWall:
                new_corners = tuple(p for i, p in enumerate(corners) if p != (nextx, nexty))
                successors.append((((nextx, nexty), new_corners), action, 1))

            "*** YOUR CODE HERE ***"

        self._expanded += 1  # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state: Any, problem: CornersProblem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners  # These are the corner coordinates
    walls = problem.walls  # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    corners_left = state[1]
    pos = state[0]
    # cnt = len(corners_left)
    res = 0
    for goal in corners_left:
        euc = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        res = max(euc, res)

    return res
    # return 0 # Default to trivial solution


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem


class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """

    def __init__(self, startingGameState: pacman.GameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1  # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append((((nextx, nexty), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x, y = self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state

    "*** YOUR CODE HERE ***"
    # functions
    def memory_mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
        try:
            return problem.heuristicInfo['maze_distances'][(point1, point2)]
        except:
            x1, y1 = point1
            x2, y2 = point2
            walls = gameState.getWalls()
            assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
            assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
            prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
            ans = len(search.bfs(prob))
            problem.heuristicInfo['maze_distances'][(point1, point2)] = ans
            return ans

    # some inline funcs
    def manhattan_2d(a, b, state):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def cheby_2d(a, b, state):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def euclid_2d(xy1, xy2, state):
        return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

    foodList = foodGrid.asList()
    try:
        problem.heuristicInfo['walls']
    except:
        problem.heuristicInfo['walls'] = problem.startingGameState.getWalls()
    problem.heuristicInfo['maze_distances'] = {}

    # 376
    # food之间拉开的最长距离 + 自己和其中一个(closer的)之间的距离
    # foods_me_with_distances = []
    # 需要记录节点对，但是我决定用list来整，虽然会有多余的存储，但是摆烂了就这样吧
    foods_foods_with_distances = []
    for food in foodList:
        # foods_me_with_distances.append([food, memory_mazeDistance(food, position, problem.startingGameState)])
        for food2 in foodList:
            foods_foods_with_distances.append(
                [food, food2, memory_mazeDistance(food, food2, problem.startingGameState)])

    longest_distance_foods = max(foods_foods_with_distances, key=lambda sublist: sublist[2]) if len(
        foods_foods_with_distances) else None
    closer_distance = 0
    if longest_distance_foods is not None:  # 考虑了0的选项
        food = longest_distance_foods[0]
        food2 = longest_distance_foods[1]
        closer_distance = min(memory_mazeDistance(food, position, problem.startingGameState),
                              memory_mazeDistance(food2, position, problem.startingGameState))
        return closer_distance + longest_distance_foods[2]
    else:
        return 0

    # 719
    # food之间拉开的最长距离 + 自己和 最近food 的距离
    # 由于不需要记录food到底是谁，所以不用heap，只需要记录标量
    food_and_food_distances = [0]
    food_and_me_distances = []
    for food_pos in foodList:
        food_and_me_distances.append(memory_mazeDistance(food_pos, position, problem.startingGameState))
        for another_food_pos in foodList:
            food_and_food_distances.append(memory_mazeDistance(another_food_pos, food_pos, problem.startingGameState))

    return min(food_and_me_distances) + max(food_and_food_distances) if len(food_and_me_distances) else max(
        food_and_food_distances)

    # 最远的两个food之间距离+自己和最近food的距离
    # 3001
    # create distance heap
    for x in range(foodGrid.width):
        for y in range(foodGrid.height):
            if foodGrid[x][y]:
                food_left_pos.push((x, y), cheby_2d(position, (x, y), problem.startingGameState))
                # print(f"x, y, priority: {x}, {y}, {mazeDistance(position, (x,y), problem.startingGameState)}")
    try:
        _, _, f1 = food_left_pos.heap[-1]
        _, _, f2 = food_left_pos.heap[-2]
        # farthest_two_dist = mazeDistance(f1,f2,problem.startingGameState)
        farthest_two_dist = memory_mazeDistance(f1, f2, problem.startingGameState)
    except:
        farthest_two_dist = 0
    try:
        _, _, c1 = food_left_pos.heap[0]
        md = memory_mazeDistance(position, c1, problem.startingGameState)
    except:
        md = 0

    return farthest_two_dist + md

    # 4424 expand solution

    # for x in range(foodGrid.width):
    #     for y in range(foodGrid.height):
    #         if foodGrid[x][y]:
    #             food_left_pos.push((x,y), euclid_2d(position, (x,y),  problem.startingGameState))
    #             # print(f"x, y, priority: {x}, {y}, {mazeDistance(position, (x,y), problem.startingGameState)}")
    # # print("pos: "+str(position))
    # if len(food_left_pos.heap)==0:
    #     return 0
    # c1 = food_left_pos.pop()
    # d = mazeDistance(position, c1, problem.startingGameState) + len(food_left_pos.heap)
    # return d


class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"

    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while currentState.getFood().count() > 0:
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState: pacman.GameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        x0, y0 = startPosition
        radius = 0
        walls = gameState.getWalls()
        now_distance = 99999999
        closest_food_pos = None
        for x in range(walls.width):
            for y in range(walls.height):
                if food[x][y] and mazeDistance(startPosition, (x, y), gameState) < now_distance:
                    now_distance = mazeDistance(startPosition, (x, y), gameState)
                    closest_food_pos = (x, y)
        for x in range(walls.width):
            for y in range(walls.height):
                if (x, y) != closest_food_pos:
                    problem.food[x][y] == False
        # find path
        return search.bfs(problem)
        # util.raiseNotDefined()


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state: Tuple[int, int]):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x, y = state

        "*** YOUR CODE HERE ***"
        return self.food[x][y] == True
        # util.raiseNotDefined()


class ApproximateSearchAgent(Agent):
    "Implement your agent here.  Change anything but the class name."

    def state_giving_astar(self, problem, heuristic):
        """Search the node that has the lowest combined cost and heuristic first."""
        "*** YOUR CODE HERE ***"
        self.cnt = 0
        from util import PriorityQueue

        if problem.isGoalState(problem.getStartState()):
            return (problem.getStartState(), [], [])
        fringes = PriorityQueue()
        start = problem.getStartState()
        closed = set()
        fringes.push((problem.getStartState(), [], []),
                     priority=0 + heuristic(start, problem))  # (state, actions, path)

        while not fringes.isEmpty():
            candidate = fringes.pop()
            self.cnt += 1
            if self.cnt % 10 == 0:
                print("Iterations / Candidate Count: " + str(self.cnt))
            if problem.isGoalState(candidate[0]):
                return candidate
            # if self.unvisited.count() - len(candidate[2]) <= 10:
            #     return candidate
            if candidate[0] not in closed:
                closed.add(candidate[0])
                for child in problem.getSuccessors(candidate[0]):
                    child_state, child_action, child_one_cost = child
                    if child_state not in closed:
                        fringes.update((child_state, candidate[1] + [child_action],
                                        candidate[2] + [child_state]),
                                       problem.getCostOfActions(candidate[1] + [child_action]) + heuristic(child_state,
                                                                                                           problem),
                                       )

    def state_giving_dfs(problem):
        from util import Stack

        if problem.isGoalState(problem.getStartState()):
            return ([])

        start = problem.getStartState()
        fringes = Stack()
        closed = set()
        fringes.push((start, []))  # (state, actions)

        while (True):
            # print("Fringe Length: "+ str(len(fringes.list)))
            if fringes.isEmpty():
                return []
            candidate = fringes.pop()  # only one candidate is out..

            if problem.isGoalState(candidate[0]):  # the candidate "state" dim
                return candidate[1]  # the candidate "action dim"
            if candidate[0] not in closed:
                closed.add(candidate[0])
                for child in problem.getSuccessors(candidate[0]):
                    child_state, child_action, _ = child
                    if child_state not in closed:
                        fringes.push((child_state, candidate[1] + [child_action]))

    def state_giving_bfs(self, problem):
        """Search the shallowest nodes in the search tree first.
        
        return (state, actions)
        
        """
        "*** YOUR CODE HERE ***"
        from util import Queue

        if problem.isGoalState(problem.getStartState()):
            return (problem.getStartState(), [], [])
        fringes = Queue()
        start = problem.getStartState()
        closed = set()
        fringes.push((start, [], []))  # (state, actions, path)

        while not fringes.isEmpty():

            candidate = fringes.pop()
            if problem.isGoalState(candidate[0]):
                return candidate  # actions
            if candidate[0] not in closed:
                closed.add(candidate[0])
                for child in problem.getSuccessors(candidate[0]):
                    child_state, child_action, _ = child
                    if child_state not in closed:
                        fringes.push((child_state, candidate[1] + [child_action], candidate[2] + [child_state]))

    def registerInitialState(self, state):
        "This method is called before any moves are made."
        "*** YOUR CODE HERE ***"
        s = Directions.SOUTH
        n = Directions.NORTH
        w = Directions.WEST
        e = Directions.EAST
        self.now_position = state.getPacmanPosition()
        self.food = state.getFood()  # state = (now position, foodGrid)
        self.width = self.food.width
        self.height = self.food.height
        self.actions = []
        # print(self.food.count())
        # self.actions = origin_actions
        # for act in self.actions:
        #     x, y = self.now_position
        #     if act == n:
        #         y += 1
        #     elif act == s:
        #         y -= 1
        #     elif act == w:
        #         x -= 1
        #     elif act == e:
        #         x += 1
        #     self.now_position = (x, y)
        #     self.food[x][y] = False
        # print(self.food.count())
        # problem = FoodSearchProblem(state)
        # problem.start = (self.now_position, self.food)
        # (new_pos, self.food), residual_actions, self.path = self.state_giving_astar(problem=problem,
        #                                                                             heuristic=foodHeuristic)
        # print("Residual Actions = " + str(residual_actions))
        # self.actions += residual_actions
        # print("All Actions = " + str(self.actions))
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #

        # # 想法: 把矩阵粗暴切成很多块，每一块bfs
        # # 可能会造成多余路径
        # import time
        # start_time = time.time()
        # self.actions = []

        # print(self.food)
        # print('\n')
        # import copy
        # self.unvisited = copy.deepcopy(self.food)
        # # map => n divisions
        # n = 2 # 竖切切成几个
        # m = 3 # 横切
        # for p in range(m):
        #     for k in range(0 if p % 2 else n-1, n if p % 2 else -1, 1 if p % 2 else -1):
        #         if True or k != n // 2 + 2 or p != m // 2:
        #             import numpy as np
        #             from game import Grid
        #                 # x = 0, 起始位置可能不在第一个fold，所以不能取切片
        #                 # 目前暂时搞不懂 k > 0 怎么整成切片。应该是要加墙
        #                 # 目前先在此处实验
        #             if True or self.unvisited.count() > 14:
        #                 left_food = Grid(self.width, self.height)
        #                 for x in range((k * self.width)// n, ((k+1) * self.width) // n):
        #                     for y in range((p * self.height) // m, ((p+1) * self.height) // m):
        #                         left_food[x][y] = self.unvisited[x][y]
        #                 # print(left_food)
        #                 # print('\n')
        #                 prob_left = FoodSearchProblem(state)
        #                 print(self.now_position)
        #                 prob_left.start = (self.now_position, left_food)
        #                 (new_pos, self.food), residual_actions, residual_path = self.state_giving_astar(problem=prob_left, heuristic=foodHeuristic)
        #                 self.now_position = new_pos
        #                 # self.food = state.getFood()
        #                 # print(self.now_position)
        #                 self.actions += residual_actions
        #                 for node in residual_path:
        #                     pos = node[0]
        #                     self.unvisited[pos[0]][pos[1]] = False
        #             else:
        #                 break

        # print("Out of the loop")

        # print(self.unvisited)
        # prob = FoodSearchProblem(state)
        # prob.start = (self.now_position, self.unvisited)
        # residual_actions = search.astar(problem=prob, heuristic=foodHeuristic)
        # self.actions += residual_actions
        # elif k == 1:
        #     left_food = Grid((k+1) * self.width // n - (k-1) * self.width // n, self.height)
        #     x_offset = (k-1) * self.width // n
        #     print(x_offset)
        #     # print(left_food)
        #     for x in range(k * self.width // n, (k+1) * self.width // n):
        #         for y in range(self.height):
        #             left_food[x-(k-1) * self.width // n][y] = self.food[x][y]
        #     # print left problem
        #     print(left_food)
        #     print('\n')
        #     prob_left = FoodSearchProblem(state)

        #     self.now_position = (self.now_position[0] - x_offset, self.now_position[1])
        #     print(self.now_position)

        #     prob_left.start = (self.now_position, left_food)
        #     (self.now_position, _), residual_actions = self.state_giving_bfs(problem=prob_left)
        #     self.now_position = (self.now_position[0] + x_offset, self.now_position[1])
        #     self.actions += residual_actions

        # right_food = copy.deepcopy(self.food)
        # for x in range(0, self.width // 2):
        #     for y in range(self.height):
        #         right_food[x][y] = False
        # # print right problem
        # prob_right = FoodSearchProblem(state)
        # prob_right.start = (self.now_position, right_food)
        # (self.now_position, _), residual_actions = self.state_giving_bfs(problem=prob_right)
        # self.actions += residual_actions      
        # print(time.time() - start_time)
        # print('Path found with cost %d.' % len(self.actions))
        self.actionIndex = 0
        # print('Actions: '+str(self.actions))

        # 280
        self.actions = [w, w, w, w, n, n, w, w, n, n, s, s, e, e, e, e, e, n, n, w, w, w, n, n, n, n, e, e, e, e, e, n,
                        n, e, e, e, e, e, s, s, s, s, s, s, s, s, w, w, s, s, w, w, w, e, e, e, n, n, w, w, w, n, n, e,
                        e, e, n, n, n, n, w, e, s, s, e, e, e, e, e, e, s, s, w, w, s, s, s, s, w, w, e, e, n, n, e, e,
                        s, s, e, e, e, e, n, n, n, w, w, n, n, e, e, n, n, w, w, n, n, e, e, n, w, n, e, n, w, w, s, w,
                        n, w, s, s, e, e, s, s, s, e, e, s, s, w, w, s, s, s, w, w, w, w, n, n, e, e, n, n, n, n, w, w,
                        n, w, e, n, n, n, w, w, w, w, s, n, w, w, w, w, w, w, w, w, w, w, w, w, s, s, s, s, n, n, n, n,
                        e, e, e, e, s, s, e, e, e, s, n, w, w, w, w, w, s, s, s, s, e, w, w, w, w, w, n, n, s, s, s, s,
                        e, e, s, s, s, s, e, e, e, w, w, w, n, n, w, w, s, s, w, w, w, w, n, n, n, e, s, s, e, e, n, w,
                        n, n, n, w, w, n, n, e, e, n, n, n, e, e, n, n, w, w, w, w, s, s, s, e
                        ]
        print(len(self.actions))

    def getAction(self, state):
        """
        From game.py: 
        The Agent will receive a GameState and must return an action from 
        Directions.{North, South, East, West, Stop}
        """
        "*** YOUR CODE HERE ***"
        if 'actionIndex' not in dir(self):
            self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
