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
from typing import NamedTuple, Union, Any

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
import math
from searchAgents import mazeDistance

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition())
                          for ghost in newGhostStates
                          if ghost.scaredTimer == 0]

        minGhostDist = min(ghostDistances, default=100)
        if minGhostDist == 0:
            return -float('inf')

        newFoodNum = successorGameState.getNumFood()
        if newFoodNum == 0:
            return float('inf')

        curFood = currentGameState.getFood()
        foodDistances = [manhattanDistance(newPos, (x, y))
                         for x in range(curFood.width)
                         for y in range(curFood.height)
                         if curFood[x][y]]
        minFoodDist = min(foodDistances, default=0)

        return - 1 / (minGhostDist + 0.5) + 1 / (minFoodDist + 0.5)



def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions()
        successorStates = [gameState.generateSuccessor(0, action) for action in actions]
        values = [self.miniMax(1, self.depth, successorState) for successorState in successorStates]
        maxValue = max(values)
        index = [i for i in range(len(values)) if values[i] == maxValue]
        return actions[index[0]]

    def miniMax(self, agentIndex, depth, gameState):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIndex)
        successorStates = [gameState.generateSuccessor(agentIndex, action) for action in actions]

        if agentIndex == 0:
            return max([self.miniMax(1, depth, successorState) for successorState in successorStates])

        else:
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth - 1 if nextAgentIndex == 0 else depth
            return min([self.miniMax(nextAgentIndex, nextDepth, successorState) for successorState in successorStates])


class ScoredAction(NamedTuple):
    score: Union[int, float]
    action: Any

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        
        "*** YOUR CODE HERE ***"
        return self.AlphaBetaPruning(0, self.depth, gameState, -float('inf'), float('inf'))[1]

    def AlphaBetaPruning(self, agentIndex, depth, gameState, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        if agentIndex == 0:
            return self.maxValue(agentIndex, depth, gameState, alpha, beta)
        else:
            return self.minValue(agentIndex, depth, gameState, alpha, beta)

    def maxValue(self, agentIndex, depth, gameState, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)

        maxAction = None
        maxValue = -float('inf')

        for action in actions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = self.AlphaBetaPruning(1, depth, successorState, alpha, beta)[0]
            if value > beta:
                return value, action
            if value > maxValue:
                maxValue = value
                maxAction = action
            alpha = max(value, alpha)
        return maxValue, maxAction

    def minValue(self, agentIndex, depth, gameState, alpha, beta):
        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgentIndex == 0 else depth
        actions = gameState.getLegalActions(agentIndex)

        minValue = float('inf')
        minAction = None

        for action in actions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = self.AlphaBetaPruning(nextAgentIndex, nextDepth, successorState, alpha, beta)[0]
            if value < alpha:
                return value, action
            if value < minValue:
                minValue = value
                minAction = action
            beta = min(beta, value)
        return minValue, minAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.Expectimax(0, self.depth, gameState)[1]

    def Expectimax(self, agentIndex, depth, gameState):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        if agentIndex == 0:
            return self.maxValue(agentIndex, depth, gameState)
        else:
            return self.expValue(agentIndex, depth, gameState)

    def maxValue(self, agentIndex, depth, gameState):
        actions = gameState.getLegalActions(agentIndex)
        maxValue = -float('inf')
        maxAction = None

        for action in actions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = self.Expectimax(1, depth, successorState)[0]
            if value > maxValue:
                maxValue = value
                maxAction = action
        return maxValue, maxAction

    def expValue(self, agentIndex, depth, gameState):
        actions = gameState.getLegalActions(agentIndex)
        expValue = 0

        nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth - 1 if nextAgentIndex == 0 else depth

        for action in actions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            expValue += 1 / len(actions) * self.Expectimax(nextAgentIndex, nextDepth, successorState)[0]

        return expValue, None

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    currentScores = currentGameState.getScore()
    if currentGameState.isWin() or currentGameState.isLose():
        return currentScores

    food = currentGameState.getFood()
    foodPos = food.asList()
    foodPos = sorted(foodPos, key=lambda pos: mazeDistance(pacmanPos, pos, currentGameState))
    mazeFoodDistances = [mazeDistance(pacmanPos, pos, currentGameState) for pos in foodPos]
    foodProfit = sum(10 / (distance + 0.5) for distance in mazeFoodDistances[:5])

    ghostStates = currentGameState.getGhostStates()
    mazeGhostDistances = [mazeDistance(pacmanPos, tuple(map(int, ghost.getPosition())),
                          currentGameState) for ghost in ghostStates]
    scaredTimes = [ghost.scaredTimer for ghost in ghostStates]
    mazeNoScaredDistances = [distance for distance, scaredTime in
                             zip(mazeGhostDistances, scaredTimes) if scaredTime == 0]
    mazeScaredDistances = [distance for distance, scaredTime in zip(mazeGhostDistances, scaredTimes) if scaredTime > 2]

    ghostDanger = -sum((300 / mazeNoScaredDistance ** 2 for mazeNoScaredDistance in mazeNoScaredDistances), 0)
    ghostProfit = sum((150 / mazeScaredDistance for mazeScaredDistance in mazeScaredDistances), 0)

    capsule = currentGameState.getCapsules()
    mazeCapsuleDistances = [mazeDistance(pacmanPos, capsulePos, currentGameState) for capsulePos in capsule]
    capsuleProfit = sum(20 / (distance ** 2 + 0.5) for distance in mazeCapsuleDistances)

    return ghostDanger + foodProfit + ghostProfit + capsuleProfit + currentScores


# Abbreviation
better = betterEvaluationFunction
