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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        pos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = [ghost for ghost in successorGameState.getGhostStates() if not ghost.scaredTimer]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        capsules = currentGameState.getCapsules()
        "*** YOUR CODE HERE ***"
        min_ghost_distance = 4
        min_food_distance = 40
        min_capsule_distance = 12

        for ghost in newGhostStates:
            min_ghost_distance = min(min_ghost_distance,
                                     abs(pos[0] - ghost.getPosition()[0]) + abs(pos[1] - ghost.getPosition()[1]))
        for food in newFood:
            min_food_distance = min(min_food_distance, abs(food[0] - pos[0]) + abs(food[1] - pos[1]))
        for capsule in capsules:
            min_capsule_distance = min(min_capsule_distance, abs(capsule[0] - pos[0]) + abs(capsule[1] - pos[1]))
        food_diff = currentGameState.getNumFood() - successorGameState.getNumFood()
        #print(successorGameState.getScore(), min_ghost_distance, (1.0 / (min_ghost_distance + .01)) * 1.5, food_diff,
        #      (1.0 / min_food_distance) * 5)
        #print("min food distance is", min_food_distance)
        #print("score is", -(1.0 / (min_ghost_distance + .01) * 4) + (1.0 / min_food_distance) * 5 + 5 * food_diff)
        # return 0
        return (-(1.0 / (min_ghost_distance + .01) * 4) - (
            min_food_distance) + 5 * food_diff) + successorGameState.getScore()  # *1000


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

    def recurse_minimax(self, gameState, current_depth):
        if current_depth == 0:
            return self.evaluationFunction(gameState)

        num_agents = gameState.getNumAgents()
        which_agent = -current_depth % num_agents

        actions = gameState.getLegalActions(which_agent)
        # if this is the case then it is a game_state
        if len(actions) == 0:
            return gameState.getScore()
        if which_agent != 0:
            [self.recurse_minimax(gameState.generateSuccessor(which_agent, action), current_depth - 1) for action in
             actions]

            return min(
                [self.recurse_minimax(gameState.generateSuccessor(which_agent, action), current_depth - 1) for action in
                 actions])
        else:  # its pacman
            return max(
                [self.recurse_minimax(gameState.generateSuccessor(which_agent, action), current_depth - 1) for action in
                 actions])

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
        """
        "*** YOUR CODE HERE ***"
        max_score = -1000000
        max_action = None
        actions = gameState.getLegalActions(0)
        for action in actions:
            val = self.recurse_minimax(gameState.generateSuccessor(0, action),
                                       gameState.getNumAgents() * self.depth - 1)
            if val > max_score:
                max_score = val
                max_action = action

        return max_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def recurse_ab(self, gameState, current_depth, alpha, beta):
        if current_depth == 0:
            return self.evaluationFunction(gameState)

        num_agents = gameState.getNumAgents()
        which_agent = -current_depth % num_agents

        actions = gameState.getLegalActions(which_agent)
        # if this is the case then it is a game_state
        if len(actions) == 0:
            return gameState.getScore()
        # minimizer ghosts
        if which_agent != 0:
            v = 100000000
            for action in actions:
                v = min(v, self.recurse_ab(gameState.generateSuccessor(which_agent, action), current_depth - 1, alpha,
                                           beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
        else:  # its pacman
            v = -100000000
            for action in actions:
                v = max(v, self.recurse_ab(gameState.generateSuccessor(which_agent, action), current_depth - 1, alpha,
                                           beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        max_score = -10000000
        alpha = -1000000000
        beta = 10000000000
        max_action = None
        actions = gameState.getLegalActions(0)
        for action in actions:
            val = self.recurse_ab(gameState.generateSuccessor(0, action), gameState.getNumAgents() * self.depth - 1,
                                  alpha, beta)
            if val > max_score:
                max_action = action
                max_score = val
            if max_score > beta:
                return action
            alpha = max(alpha, val)

        return max_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def recurse_expectimax(self, gameState, current_depth):
        if current_depth == 0:
            return self.evaluationFunction(gameState)

        num_agents = gameState.getNumAgents()
        which_agent = -current_depth % num_agents

        actions = gameState.getLegalActions(which_agent)
        # if this is the case then it is a game_state
        if len(actions) == 0:
            return gameState.getScore()
        if which_agent != 0:
            arr = [self.recurse_expectimax(gameState.generateSuccessor(which_agent, action), current_depth - 1) for
                   action in actions]
            return (sum(arr) + 0.0) / len(arr)
        else:  # its pacman
            return max(
                [self.recurse_expectimax(gameState.generateSuccessor(which_agent, action), current_depth - 1) for action
                 in actions])

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        max_score = -1000000
        max_action = None
        actions = gameState.getLegalActions(0)
        for action in actions:
            val = self.recurse_expectimax(gameState.generateSuccessor(0, action),
                                          gameState.getNumAgents() * self.depth - 1)
            if val > max_score:
                max_score = val
                max_action = action
        return max_action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    # Useful information you can extract from a GameState (pacman.py)
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    min_ghost_distance = 4
    min_food_distance = 40
    min_capsule_distance = 12
    for ghost in ghostStates:
        min_ghost_distance = min(min_ghost_distance,
                                 abs(pos[0] - ghost.getPosition()[0]) + abs(pos[1] - ghost.getPosition()[1]))
    for food in foods:
        min_food_distance = min(min_food_distance, abs(food[0] - pos[0]) + abs(food[1] - pos[1]))
    for capsule in capsules:
        min_capsule_distance = min(min_capsule_distance, abs(capsule[0] - pos[0]) + abs(capsule[1] - pos[1]))

    return (-(1.0 / (min_ghost_distance + .01) * 4) - (
        min_food_distance) - 5*len(foods)) + currentGameState.getScore() - 10*len(capsules)  # *1000


# Abbreviation
better = betterEvaluationFunction
