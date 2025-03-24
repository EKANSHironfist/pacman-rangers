import random
from captureAgents import CaptureAgent
from game import Directions
from dataclasses import dataclass
import math
import time
from config import *
import util

class MCTSAgent(CaptureAgent):
    """
    A simple agent that picks a random legal action.
    """

    def registerInitialState(self, gameState):
        """
        Initialize the agent at the start of the game.
        """
        CaptureAgent.registerInitialState(self, gameState)

    def evaluate_off(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_off_features(gameState, action)
        weights = self.get_off_weights(gameState, action)
        return features * weights

    def get_off_features(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        next_tate = self.get_next_state(gameState, action)
        if next_tate.getAgentState(self.index).numCarrying > gameState.getAgentState(self.index).numCarrying:
            features['getFood'] = 1
        else:
            if len(self.getFood(next_tate).asList()) > 0:
                features['minDistToFood'] = self.get_min_dist_to_food(next_tate)
        return features

    def get_off_weights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'minDistToFood': -1, 'getFood': 100}

    def evaluate_def(self, gameState, action, ghosts):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_def_features(gameState, action)
        weights = self.get_def_weights(gameState, action)
        return features * weights

    def get_def_features(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_next_state(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList) 

        if len(foodList) > 0:
            current_pos = successor.getAgentState(self.index).getPosition()
            min_distance = min([self.getMazeDistance(current_pos, food) for food in foodList])
            features['distanceToFood'] = min_distance
        return features

    def get_def_weights(self, gameState, action):

        return {'successorScore': 100, 'distanceToFood': -1}

    def get_next_state(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def get_min_dist_to_food(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        return min([self.getMazeDistance(myPos, f) for f in self.getFood(gameState).asList()])

    def detect_enemy_border(self, gameState):
        """
        Return borders position
        """
        walls = gameState.getWalls().asList()
        if self.red:
            border_x = self.arena_width // 2
        else:
            border_x = self.arena_width // 2 - 1
        border_line = [(border_x, h) for h in range(self.arena_height)]
        return [(x, y) for (x, y) in border_line if (x, y) not in walls and (x + 1 - 2*self.red, y) not in walls]

    def detect_enemy_ghost(self, gameState):
        """
        Return Observable Oppo-Ghost Index
        """
        enemyList = []
        for enemy in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(enemy)
            if (not enemyState.isPacman) and enemyState.scaredTimer == 0:
                enemyPos = gameState.getAgentPosition(enemy)
                if enemyPos != None:
                    enemyList.append(enemy)
        return enemyList

    def detect_enemy_approaching(self, gameState):
        """
        Return Observable Oppo-Ghost Position Within 5 Steps
        """
        dangerGhosts = []
        ghosts = self.detect_enemy_ghost(gameState)
        myPos = gameState.getAgentPosition(self.index)
        for g in ghosts:
            distance = self.getMazeDistance(myPos, gameState.getAgentPosition(g))
            if distance <= 5:
                dangerGhosts.append(g)
        return dangerGhosts

    def detect_enemy_pacman(self, gameState):
        """
        Return Observable Oppo-Pacman Position
        """
        enemyList = []
        for enemy in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(enemy)
            if enemyState.isPacman and gameState.getAgentPosition(enemy) != None:
                enemyList.append(enemy)
        return enemyList

    def chooseAction(self, gameState):

        legal_actions = gameState.getLegalActions(self.index)
        agent_state = gameState.getAgentState(self.index)

        #Use MCTS is enemy agent is detected and use heuristics otherwise
        if gameState.getAgentState(self.index).isPacman: #when on enemy side - going offensive
            food_pellets = self.getFood(gameState).asList()
            enemy_ghost_detected = [gameState.getAgentPosition(enemy) for enemy in self.getOpponents(gameState) if ((gameState.getAgentPosition(enemy)!=None) and (not gameState.getAgentState(enemy).isPacman))]

            if not enemy_ghost_detected:
                values = [self.evaluate_off(gameState, a) for a in legal_actions]
                maxValue = max(values)
                bestActions = [a for a, v in zip(legal_actions, values) if v == maxValue]
                return random.choice(bestActions)
            else:    
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
                    reward = node.rollout(max_depth=rollout_depth,epsilon=epsilon, revisit_penalty=revisit_penalty) # Simulate rollout
                    node.backpropagate(reward) # Backpropagate the reward to the upper nodes of the tree
            
                return root_node.best_action() # Return the best action

        else: #when at homebase defending
            ghosts = self.detect_enemy_ghost(gameState)
            values = [self.evaluate_def(gameState, a, ghosts) for a in legal_actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(legal_actions, values) if v == maxValue]
            return random.choice(bestActions)

class DefensiveAgent(MCTSAgent):
    pass


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

    def ucb1(self, Cp=1/math.sqrt(2), revisit_penalty=3):
        n_parent = self.parent.visits
        n_j = self.visits
        if n_j==0:
            return float('inf')
        avg_reward = self.total_reward/n_j
        backtrack_penalty = revisit_penalty*self.position_history.count(self.position)
        score = avg_reward + Cp * math.sqrt(math.log(n_parent)/n_j) - backtrack_penalty

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
    
    def rollout(self, max_depth=20, epsilon=0.7, revisit_penalty=2):
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
                        score -= revisit_penalty*visited_positions.count(next_pos)
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
