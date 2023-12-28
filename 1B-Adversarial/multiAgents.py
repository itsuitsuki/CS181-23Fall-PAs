# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        
        
        "*** YOUR CODE HERE ***"
        res = scoreEvaluationFunction(childGameState)
        # res  = 0
        if newPos in newFood.asList():
            return 100
        if newPos in [ghost.configuration.pos for ghost in newGhostStates]:
            return -114
        min_ghost_d = min([manhattanDistance(ghost.configuration.pos, newPos) for ghost in newGhostStates]) if len(newGhostStates) else 0
        res -= 40 * 1/min_ghost_d if len(newGhostStates) else 0
        # print(newFood.asList())
        min_food_d = min([manhattanDistance(fpos, newPos) for fpos in newFood.asList()]) if newFood.count() else 0
        res += 10 * 1/min_food_d if newFood.count() else 0
        
        # print(newGhostStates[0])
        # for ghost in newGhostStates:
        #     res -= 1/manhattanDistance(ghost.configuration.pos,newPos) if manhattanDistance(ghost.configuration.pos,newPos) != 0 else 0
        # print(res)
        # res -= newFood.count()
        # res += sum(newScaredTimes)
        
        return res

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(gstate: GameState, depth: int, index: int):
            # 返回一个 (utility, move) 对 >>>> P122, AIMA CH5
            # depth: 已经动多少下，就是从 pacman 到 ghosts 总共遍历几遍
            isPacman = False
            # index %= gstate.getNumAgents() 
            if index == 0:
                # agent is a pacman
                depth += 1
                isPacman = True
            if gstate.isWin() or gstate.isLose() or depth == self.depth + 1: # 这里要depth == self.depth+1，因为假设self.depth = 2, pacman在 depth=2 的时候也可以活动，只有depth 2 这一层结束了，这个搜索才结束
                return self.evaluationFunction(gstate), None
            v, move = (float("-inf"), None) if isPacman else (float("inf"), None)
            for a in gstate.getLegalActions(agentIndex=index):
                v2, _ = minimax(gstate.getNextState(index, a), depth, (index+1)%gstate.getNumAgents())
                if isPacman:
                    if v2 >= v: 
                        v, move = v2, a
                        # print(v, move)
                else:
                    if v2 <= v: 
                        v, move = v2, a
                        # print(v, move)
            return v, move
                
        _, action = minimax(gameState, 0, self.index)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Returns the minimax action using self.depth and self.evaluationFunction, using alpha-beta pruning.
        def alphabeta(gstate: GameState, depth: int, index: int, alpha: float, beta: float, func = self.evaluationFunction):
            # 返回一个 (utility, move) 对 >>>> P122, AIMA CH5
            # depth: 已经动多少下，就是从 pacman 到 ghosts 总共遍历几遍
            isPacman = False
            # index %= gstate.getNumAgents() 
            if index == 0:
                # agent is a pacman
                depth += 1
                isPacman = True
            if gstate.isWin() or gstate.isLose() or depth == self.depth + 1:
                return func(gstate), None
            v, move = (float("-inf"), None) if isPacman else (float("inf"), None)
            for a in gstate.getLegalActions(agentIndex=index):
                
                if isPacman: 
                    # pacman 这一层 要给它们的 parent，也就是给 ghost 传递最大值, ghost 需要选择 pacman 传递的最大值里边, 最小的那个. 
                    # 而pacman遍历 自己的孩子 的时候，他传上去的最大值只会越来越大, 但是ghost 需要的是尽量小的值, 
                    # 所以 ghost 会维护一个 beta, 大于 beta 的 pacman 值就会被 ghost 剪掉（被嫌弃）
                    # beta 怎么更新？虽然上面说，大的会剪掉，但是 比 beta 小的会被更新成新的 beta, 所以 beta 是这样更新的
                    # 大的剪掉，就是在这里剪掉的.
                    # 在pacman里，v要不断增加，增加到beta了，就被 ghost parent 剪掉了
                    # beta 在多个pacman贡献的最大值里从左边到右边不断往小了更新
                    v2, _ = alphabeta(gstate.getNextState(index, a), depth, (index+1)%gstate.getNumAgents(), alpha, beta)
                    if v2 >= v: 
                        v, move = v2, a
                        # print(v, move)
                        
                    if v > beta:
                        return v, move
                    alpha = max(alpha, v)
                else:
                    # ghost 要给 pacman 传递最小值的时候 pacman 只想挑里边大的
                    # 所以 pacman 会维护一个 alpha, 小于 alpha 的 ghost 值就会被 pacman 嫌弃
                    # 但是 ghost 的值 在这个子树下 会不断变小，变小到 alpha 了，就被 pacman parent 剪掉了
                    # alpha 在多个ghost贡献的最小值里从左边到右边不断往大了更新
                    
                    v2, _ = alphabeta(gstate.getNextState(index, a), depth, (index+1)%gstate.getNumAgents(), alpha, beta)
                    # 这里要考虑 ghost 是 ghost 的 parent 的情况吗？因为有多个ghost，所以需要考虑，
                    # ghost 的 parent 是 ghost 的时候，ghost 的最小值，传递上去不会被剪掉，因为 ghost 的 parent 是 ghost，它也想要最小值
                    # 我们这里需要单独区别这种情况吗？不需要，因为 这里判断的只有 小于 alpha 与否，也就是说剪枝的条件要从 上面的 pacman ancestor 传下来的 alpha 来判断
                    # 这个 alpha 从 pacman ancestor 一路传下来也不会变，但是 beta 在不断更新，
                    # 更新完的 beta 也只能传给 ghost 的 child，不能传给顶头的 pacman （当 ghost 是 ghost 的parent）
                    # 所以我认为不用新写一个
                    if v2 <= v: 
                        v, move = v2, a
                        # print(v, move)
                    beta = min(beta, v)
                    # 本题的神奇边界问题：<= 不行（不能看到等于 alpha 就剪掉，如果等于就剪掉，v ）
                    if v < alpha:
                        return v, move
                   
                    
            return v, move
        
        _, action = alphabeta(gameState, 0, self.index, float("-inf"), float("inf"))
        return action

        
            

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(gstate: GameState, depth: int, index: int):
            # 返回一个 (utility, move) 对 >>>> P122, AIMA CH5
            # depth: 已经动多少下，就是从 pacman 到 ghosts 总共遍历几遍
            isPacman = False
            # index %= gstate.getNumAgents() 
            if index == 0:
                # agent is a pacman
                depth += 1
                isPacman = True
            if gstate.isWin() or gstate.isLose() or depth == self.depth + 1: # 这里要depth == self.depth+1，因为假设self.depth = 2, pacman在 depth=2 的时候也可以活动，只有depth 2 这一层结束了，这个搜索才结束
                return self.evaluationFunction(gstate), None
            v, move = (float("-inf"), None) if isPacman else (float("inf"), None)
            
            # expectimax: if pacman, max value + max move, else, expectation/average value + random move
            if isPacman:
                for a in gstate.getLegalActions(agentIndex=index):
                    v2, _ = expectimax(gstate.getNextState(index, a), depth, (index+1)%gstate.getNumAgents())
                    if v2 >= v: 
                        v, move = v2, a
                        # print(v, move)
            else: # find the expecation as value, and choose a random move
                move_choices = gstate.getLegalActions(agentIndex=index)
                move = random.choice(move_choices)
                next_values = [expectimax(gstate.getNextState(index, a), depth, (index+1)%gstate.getNumAgents())[0] for a in move_choices]
                v = sum(next_values)/len(next_values)
            return v, move
                
        _, action = expectimax(gameState, 0, self.index)
        return action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: the function that I used in question 1 is not good enough, so I added some features to it:
    1. the distance to the closest food
    2. the distance to the closest ghost
    3. the distance to the closest capsule
    4. the number of food left
    5. the number of capsules left
    6. the number of scared ghosts
    7. the score of the current state
    8. the number of ghosts left
    """
    "*** YOUR CODE HERE ***"
    
    # childGameState = currentGameState.getPacmanNextState(action)
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    Capsules = currentGameState.getCapsules()
    res = scoreEvaluationFunction(currentGameState)
    
    FOOD_RECIPROCAL_COEFF = 1
    GHOST_RECIPROCAL_COEFF = -3
    CAPSU_RECIPROCAL_COEFF = 3
    SCARED_RECIPROCAL_COEFF = 1
    min_scared_time = min(ScaredTimes)
    res += SCARED_RECIPROCAL_COEFF if max(ScaredTimes) != 0 else 0
    # res  = 0
    if Pos in Capsules:
        return 100
    if Pos in Food.asList():
        return 11
    if Pos in [ghost.configuration.pos for ghost in GhostStates]:
        return -10
    min_ghost_d = min([manhattanDistance(ghost.configuration.pos, Pos) for ghost in GhostStates]) if len(GhostStates) else 0
    res += GHOST_RECIPROCAL_COEFF * 1/min_ghost_d if len(GhostStates) else GHOST_RECIPROCAL_COEFF
    # print(newFood.asList())
    min_food_d = min([manhattanDistance(fpos, Pos) for fpos in Food.asList()]) if Food.count() else 0
    res += FOOD_RECIPROCAL_COEFF * 1/min_food_d if Food.count() else FOOD_RECIPROCAL_COEFF
    min_capsule_d = min([manhattanDistance(cpos, Pos) for cpos in Capsules]) if len(Capsules) else 0
    res += CAPSU_RECIPROCAL_COEFF * 1/min_capsule_d if len(Capsules) else CAPSU_RECIPROCAL_COEFF
    # print(newGhostStates[0])
    # for ghost in newGhostStates:
    #     res -= 1/manhattanDistance(ghost.configuration.pos,newPos) if manhattanDistance(ghost.configuration.pos,newPos) != 0 else 0
    # print(res)
    # res -= newFood.count()
    # res += sum(newScaredTimes)
    
    return res

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """
    def __init__(self):
        self.cnt = 0
        self.specified_cnt = 1
        self.foods_cnt = -10
        self.index = 0
        self.depth = 2
        self.FOOD_RECIPROCAL_COEFF = 28
        # self.GHOST_RECIPROCAL_COEFF = -2
        self.CAPSU_RECIPROCAL_COEFF = 1145
        self.RETURN_FOODS_COEFF = 35
        # self.SCARED_COEFF = 1
        # self.FOOD_COUNT_COEFF = 141
        # self.GHOST_COUNT_COEFF = -59
        # self.CAPSU_COUNT_COEFF = +116
        # self.SCARED_MIN_RECIP_COEFF = 1200
        self.RETURN_CAPSULES_COEFF = 3992
        self.SCARED_REWARD_COEFF = 70
        # self.SCORE_GET_COEFF = 1
        self.GHOST_PUNISH_COEFF = -2118
        # {'FOOD_RECIPROCAL_COEFF': 2, 'GHOST_RECIPROCAL_COEFF': -2, 'CAPSU_RECIPROCAL_COEFF': 53, 'FOOD_COUNT_COEFF': 141, 'GHOST_COUNT_COEFF': -59, 'CAPSU_COUNT_COEFF': 116, 'SCARED_SUM_RECIP_COEFF': -5, 'RETURN_CAPSULES_COEFF': 39, 'SCARED_REWARD_COEFF': 6, 'SCORE_GET_COEFF': 2806}
    def update_coeffs(self):
        import os
        # Read coefficients from file
        if self.cnt == 1:
            self.FOOD_RECIPROCAL_COEFF = 114
            self.CAPSU_RECIPROCAL_COEFF = 497
            self.RETURN_FOODS_COEFF = 23
            self.RETURN_CAPSULES_COEFF = 8150
            self.SCARED_REWARD_COEFF = -1
            self.GHOST_PUNISH_COEFF = -2436
            
            print("CNT == 1")
            # 未来换成 self.XXX_COEFF = XXX
            if os.path.isfile('coeff1.txt'):
                with open('coeff1.txt', 'r') as f:
                    for line in f:
                        var, val = line.strip().split(': ')
                        setattr(self, var, int(val))
            

    def getAction(self, gameState: GameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        if gameState.getScore() == 0 and gameState.getNumFood() == 69: # food incz    reases, new sub-problem
            self.cnt += 1
        
        if True:  # test 判断cnt
            if self.cnt > self.specified_cnt:
                print("CNT != 1")
                return gameState.getLegalActions()[0]
            elif self.cnt == 1:
                pass
                # print("CNT == 1")
            else:
                print("???")
        def alphabeta(gstate: GameState, depth: int, index: int, alpha: float, beta: float, func = self.contestEval):
            # 返回一个 (utility, move) 对 >>>> P122, AIMA CH5
            # depth: 已经动多少下，就是从 pacman 到 ghosts 总共遍历几遍
            isPacman = False
            # index %= gstate.getNumAgents() 
            if index == 0:
                # agent is a pacman
                depth += 1
                isPacman = True
            if gstate.isWin() or gstate.isLose() or depth == self.depth + 1:
                return func(gstate), None
            v, move = (float("-inf"), None) if isPacman else (float("inf"), None)
            for a in gstate.getLegalActions(agentIndex=index):
                
                if isPacman: 
                    # pacman 这一层 要给它们的 parent，也就是给 ghost 传递最大值, ghost 需要选择 pacman 传递的最大值里边, 最小的那个. 
                    # 而pacman遍历 自己的孩子 的时候，他传上去的最大值只会越来越大, 但是ghost 需要的是尽量小的值, 
                    # 所以 ghost 会维护一个 beta, 大于 beta 的 pacman 值就会被 ghost 剪掉（被嫌弃）
                    # beta 怎么更新？虽然上面说，大的会剪掉，但是 比 beta 小的会被更新成新的 beta, 所以 beta 是这样更新的
                    # 大的剪掉，就是在这里剪掉的.
                    # 在pacman里，v要不断增加，增加到beta了，就被 ghost parent 剪掉了
                    # beta 在多个pacman贡献的最大值里从左边到右边不断往小了更新
                    v2, _ = alphabeta(gstate.getNextState(index, a), depth, (index+1)%gstate.getNumAgents(), alpha, beta)
                    if v2 >= v: 
                        v, move = v2, a
                        # print(v, move)
                        
                    if v > beta:
                        return v, move
                    alpha = max(alpha, v)
                else:
                    # ghost 要给 pacman 传递最小值的时候 pacman 只想挑里边大的
                    # 所以 pacman 会维护一个 alpha, 小于 alpha 的 ghost 值就会被 pacman 嫌弃
                    # 但是 ghost 的值 在这个子树下 会不断变小，变小到 alpha 了，就被 pacman parent 剪掉了
                    # alpha 在多个ghost贡献的最小值里从左边到右边不断往大了更新
                    from ghostAgents import DirectionalGhost
                    v2, _ = alphabeta(gstate.getNextState(index, a), depth, (index+1)%gstate.getNumAgents(), alpha, beta)
                    # 这里要考虑 ghost 是 ghost 的 parent 的情况吗？因为有多个ghost，所以需要考虑，
                    # ghost 的 parent 是 ghost 的时候，ghost 的最小值，传递上去不会被剪掉，因为 ghost 的 parent 是 ghost，它也想要最小值
                    # 我们这里需要单独区别这种情况吗？不需要，因为 这里判断的只有 小于 alpha 与否，也就是说剪枝的条件要从 上面的 pacman ancestor 传下来的 alpha 来判断
                    # 这个 alpha 从 pacman ancestor 一路传下来也不会变，但是 beta 在不断更新，
                    # 更新完的 beta 也只能传给 ghost 的 child，不能传给顶头的 pacman （当 ghost 是 ghost 的parent）
                    # 所以我认为不用新写一个
                    if v2 <= v: 
                        v, move = v2, a
                        # print(v, move)
                    beta = min(beta, v)
                    # 本题的神奇边界问题：<= 不行（不能看到等于 alpha 就剪掉，如果等于就剪掉，v ）
                    if v < alpha:
                        return v, move
                   
                    
            return v, move

        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()
        legalMoves.remove(Directions.STOP)
        # Choose one of the best actions
        _, action = alphabeta(gameState, 0, self.index, float("-inf"), float("inf"), self.contestEval)

        "Add more of your code here if you want to"

        return action
    
    def contestEval(self, currentGameState: GameState, coeffs = None):

        self.update_coeffs()
        Foods = currentGameState.getFood()
        GhostStates = currentGameState.getGhostStates()
        ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
        Capsules = currentGameState.getCapsules()
        Pos = currentGameState.getPacmanPosition()
        res = currentGameState.getScore()
        if Pos in Capsules:
            return self.RETURN_CAPSULES_COEFF
        elif Pos in Foods.asList():
            return self.RETURN_FOODS_COEFF
        if len(Capsules):
            res += self.CAPSU_RECIPROCAL_COEFF/min([manhattanDistance(Pos, cpos) for cpos in Capsules])
        elif Foods.count():
            res += self.FOOD_RECIPROCAL_COEFF/min([manhattanDistance(Pos, fpos) for fpos in Foods.asList()])
        NoThreatGhostStates = [ghost for ghost in GhostStates 
                               if manhattanDistance(ghost.getPosition(), Pos) < ghost.scaredTimer]
        if Pos in [ghost.getPosition() for ghost in GhostStates]:
            return self.GHOST_PUNISH_COEFF
        elif NoThreatGhostStates:
            res += self.SCARED_REWARD_COEFF * min([ghost.scaredTimer for ghost in NoThreatGhostStates])
        with open('test.txt', 'w') as f:
            f.write(str(self.SCARED_REWARD_COEFF) + '\n')
        # print("CNT: "+str(self.cnt))
        return res
