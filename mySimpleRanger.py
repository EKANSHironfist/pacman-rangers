import random
from captureAgents import CaptureAgent
from game import Directions

class MCTSAgent(CaptureAgent):
    """
    A simple agent that picks a random legal action.
    """

    def registerInitialState(self, gameState):
        """
        Initialize the agent at the start of the game.
        """
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Selects a random legal action.
        """
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)

def createTeam(firstIndex, secondIndex, isRed,
               first='MCTSAgent', second='MCTSAgent'):
    """
    Returns a team of two random-action agents.
    """
    return [MCTSAgent(firstIndex), MCTSAgent(secondIndex)]
