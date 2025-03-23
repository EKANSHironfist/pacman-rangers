import random
from captureAgents import CaptureAgent
from game import Directions
from dataclasses import dataclass
import math
import time
from config import *

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
        # actions = gameState.getLegalActions(self.index)
        
        # return random.choice(actions)
        start_time = time.time() # start search
        time_limit = 0.3 # time limit to perform action 
        root_node = MCTSNode(gameState, self.index) # node is defined by the game state and index see Class below
        # Perform simulations withing the defined time
        while time.time() - start_time < time_limit:
            # Select one potential node
            node = root_node.select()
            if not node:
                continue  # Prevent unnecessary expansions
            if not node.is_terminal(): # If the node is not terminal then perfrom expansion and rollout
                node = node.expand() # Expand unattempted actions
            reward = node.rollout(max_depth=rollout_depth,epsilon=epsilon) # Simulate rollout
            node.backpropagate(reward) # Backpropagate the reward to the upper nodes of the tree
        print(root_node.untried_actions)
        return root_node.best_action() # Return the best action

class MCTSNode:
    def __init__(self, gameState, agentIndex, parent=None, action=None):
        self.gameState = gameState
        self.agentIndex = agentIndex
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0 # How many times a node is visited
        self.total_reward = 0 # Reward obtained from each rollout simulation
        self.untried_actions = gameState.getLegalActions(agentIndex) # Set of unexplored actions

        #Records to track position history
        self.position = gameState.getAgentPosition(agentIndex)
        self.position_history = []
        if parent and parent.position:
            self.position_history = parent.position_history.copy()
        self.position_history.append(self.position)

    def ucb1(self, Cp=1/math.sqrt(2)):
        n_parent = self.parent.visits
        n_j = self.visits
        if n_j==0:
            return float('inf')
        avg_reward = self.total_reward/n_j
        score = avg_reward + Cp * math.sqrt(math.log(n_parent)/n_j)

        return score
    def select(self):
        if self.untried_actions:
            return self.expand()
        if not self.children:
            return self
        
        return max(self.children, key= lambda child: child.ucb1(Cp))
    
    def expand(self):
        if not self.untried_actions:
            return self
     
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        next_state = self.gameState.generateSuccessor(self.agentIndex, action)
        child_node = MCTSNode(next_state, self.agentIndex, self, action=action)
        self.children.append(child_node)
     
        return random.choice(self.children)
    
    def rollout(self, max_depth=20, epsilon=0.7):
        state = self.gameState
        depth = 0

        visited_positions = self.position_history.copy()

        while depth<max_depth:
            actions= state.getLegalActions(self.agentIndex)
            if not actions:
                break
            if random.random() < epsilon:
                scores=[]
                for action in actions:
                    next_state = state.generateSuccessor(self.agentIndex, action)
                    next_pos = next_state.getAgentPosition(self.agentIndex)
                    score = next_state.getScore()
                    if next_pos in visited_positions:
                        score -= 2*visited_positions.count(next_pos)
                    scores.append(score)
                max_score = max(scores)
                best_actions = [action for index,action in enumerate(actions) if scores[index]==max_score]
                action = random.choice(best_actions)
            else:
                action = random.choice(actions)
            
            state = state.generateSuccessor(self.agentIndex, action)
            depth+=1
            if state.isOver():
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

        return max(self.children, key=lambda child: child.total_reward/(child.visits+1e-6)).action # Choose action with highest visit count


    def is_terminal(self):
        return self.gameState.isOver()



def createTeam(firstIndex, secondIndex, isRed,
               first='MCTSAgent', second='MCTSAgent'):
    """
    Returns a team of two random-action agents.
    """
    return [MCTSAgent(firstIndex), MCTSAgent(secondIndex)]
