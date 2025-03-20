import math
import random
import time
from captureAgents import CaptureAgent
from game import Directions


class MCTSAgent(CaptureAgent):
    """

    Monte Carlo tree search 

    The agent will be using MCTS to choose the best action ,
    Expand the action that have not been attempted, simulate a random game and backpropagate
    the reward. This process will be repeated iteratively

    """
    def registerInitialState(self, gameState):
        """
        Initalize the agent at the start of the game.
        """
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        MTCS to choose the best action
        Choose best action from the simulation perfromed in a defined time
        
        """
        start_time = time.time() # start search
        time_limit = 0.1 # time limit to perform action 
        root_node = MCTSNode(gameState, self.index) # node is defined by the game state and index see Class below
        # Perform simulations withing the defined time
        while time.time() - start_time < time_limit:
            # Select one potential node
            node = root_node.select()
            if not node:
                continue  # Prevent unnecessary expansions
            if not node.is_terminal(): # If the node is not terminal then perfrom expansion and rollout
                node = node.expand() # Expand unattempted actions
            reward = node.rollout() # Simulate rollout
            node.backpropagate(reward) # Backpropagate the reward to the upper nodes of the tree

        return root_node.best_action() # Return the best action

## Class representing a MCTS node
class MCTSNode:
    """
    Parameters:
        gameState : 
        agentIndex : index of the agent using the node
        parent : parent node of current node 
        action : action performed to reach the current node so from partent -> current node 

    """
    def __init__(self, gameState, agentIndex, parent=None, action=None):
        self.gameState = gameState
        self.agentIndex = agentIndex
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0 # How many times a node is visited
        self.total_reward = 0 # Reward obtained from each rollout simulation
        self.untried_actions = gameState.getLegalActions(agentIndex) # Set of unexplored actions

    """

    """
    def ucb1(self, Cp= 1 / math.sqrt(2)):
        """
        Computes the Upper confidence bound score to select the best available node 

        Cp  = exploration factor constant 
        visits = number of itmes the parent node 
        """
        n_parent = self.parent.visits # Number of times the parent node has been visited
        n_j = self.visits # Number of times child node j has been visited
        if n_j == 0:
            return float('inf')  # Alows to explore unvisited nodes first

        avg_reward = self.total_reward / n_j # Average reward of all nodes beneath the current node 
        score = avg_reward + Cp * math.sqrt(math.log(n_parent) / n_j)

        return score
    def select(self):
        """
        Selects best child node using the UCB1 score

        When finding nodes with unattempted actions, expandes the node first
        If not , then selects the child node with higest UCB1 score
        """
        if self.untried_actions: # Evaluate if there are untried actions
            return self.expand()

        if not self.children: # Evaluate if there are no children nodes, would be the case for a terminal node
            return self  # Avoid returning None
        # Chose node with highest score
        return max(self.children, key=lambda child: child.ucb1())

    def expand(self):
        """
        Expand the tree, adding child node 

        Create a new child node, with new unattempted actions which then returns the new game state
        """
        if not self.untried_actions:
            return self  # No more possible actions
        action = self.untried_actions.pop() # Choose an untried action 
        next_state = self.gameState.generateSuccessor(self.agentIndex, action) # Generate new next state
        child_node = MCTSNode(next_state, self.agentIndex, parent=self, action=action) # Initialize new state
        self.children.append(child_node) # Add new node to the list of new
        return child_node

    def rollout(self, max_depth=5):
        """
        Simulates a random game from the current node (state)
        # The simulation rollout will be limited by max_depth to reduce computational cost
        # max_depth : change this parameter and evaluate the MCST performance

        """
        state = self.gameState # Current sate
        depth = 0 # Counter of depth
        while depth < max_depth:
            actions = state.getLegalActions(self.agentIndex) # Availabe unexplored actions
            if not actions:
                break
            action = random.choice(actions) 
            state = state.generateSuccessor(self.agentIndex, action)
            depth += 1 # Increase depth by +1
            if state.isOver():  # Stop simulation if the state is terminal or goame is over
                break
        return state.getScore()

    def backpropagate(self, reward):
        """
        Backpropage node reward values up the tree to get the score/ quality of the rollout

        """
        self.visits += 1 # Increase the visit
        self.total_reward += reward #  Accumulate reward 
        if self.parent:
            self.parent.backpropagate(reward) #if the node is a parent then iteratively propagte the reward through parent nodes unitl reaching root node

    def best_action(self):
        """
        Function to return the best action

        How to choose best action? The node-action with highest number of visits will be selected as best action
        # The better a game state is the more simulations are likely to end up winning the game
        """
        if not self.children: #If there are no children then reaching a terminal node stop
            return Directions.STOP
        return max(self.children, key=lambda child: child.visits).action # Choose action with highest visit count

    def is_terminal(self):
        """
        Checks if the current game state is a terminal state.
        """
        return self.gameState.isOver()


def createTeam(firstIndex, secondIndex, isRed,
               first='MCTSAgent', second='MCTSAgent'):
    """
    Returns a team of two MCTS-based agents.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]
