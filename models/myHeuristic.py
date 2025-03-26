from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from util import nearestPoint

#################
# Team Creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
    Returns a list of two agents forming the team.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """ A base class for reflex agents that chooses score-maximizing actions. """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        if len(self.getFood(gameState).asList()) <= 2:
            return self.returnToStart(gameState, actions)
        
        return random.choice(bestActions)
    
    def returnToStart(self, gameState, actions):
        """ Returns the best action to move towards the starting position. """
        bestDist, bestAction = float('inf'), None
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            dist = self.getMazeDistance(self.start, successor.getAgentPosition(self.index))
            if dist < bestDist:
                bestDist, bestAction = dist, action
        return bestAction
    
    def getSuccessor(self, gameState, action):
        """ Finds the next successor state. """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        return successor.generateSuccessor(self.index, action) if pos != nearestPoint(pos) else successor

    def evaluate(self, gameState, action):
        """ Computes a linear combination of features and weights. """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """ Returns a counter of features for the state. """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """ Returns feature weights. """
        return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    """ A reflex agent that seeks food. """

    def getFeatures(self, gameState, action):
        features = super().getFeatures(gameState, action)
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)

        if foodList:
            myPos = successor.getAgentState(self.index).getPosition()
            features['distanceToFood'] = min(self.getMazeDistance(myPos, food) for food in foodList)
        
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
    """ A reflex agent that prevents opponents from scoring. """

    def getFeatures(self, gameState, action):
        features = super().getFeatures(gameState, action)
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        features['onDefense'] = 0 if myState.isPacman else 1
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()]
        features['numInvaders'] = len(invaders)

        if invaders:
            features['invaderDistance'] = min(self.getMazeDistance(myPos, a.getPosition()) for a in invaders)
        
        features['stop'] = 1 if action == Directions.STOP else 0
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        features['reverse'] = 1 if action == rev else 0
        
        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
