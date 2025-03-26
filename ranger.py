import random
from captureAgents import CaptureAgent
from game import Directions
import math
import time

class MCTSAgent(CaptureAgent):
    """
    A Pacman agent using Monte Carlo Tree Search (MCTS).
    """

    def registerInitialState(self, gameState):
        """
        Initialize the agent at the start of the game.
        """
        CaptureAgent.registerInitialState(self, gameState)
        # You can add any initialization code here

    def chooseAction(self, gameState):
        """
        Uses MCTS to select an action.
        """
        # Parameters for MCTS
        start_time = time.time()
        time_limit = 0.9  # Increased time limit for better exploration
        exploration_weight = 1.5  # Slightly higher exploration parameter
        
        # Create the root node
        root_node = MCTSNode(gameState, self.index)
        num_simulations = 0
        
        # Run MCTS simulations until time limit
        while time.time() - start_time < time_limit:
            # Phase 1: Selection - traverse the tree to find a node to expand
            node = root_node
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child(exploration_weight)
            
            # Phase 2: Expansion - if the node is not terminal, expand it
            if not node.is_terminal():
                node = node.expand()
            
            # Phase 3: Simulation - perform a random playout from the node
            reward = node.rollout(max_depth=15)
            
            # Phase 4: Backpropagation - update the statistics up the tree
            node.backpropagate(reward)
            
            num_simulations += 1
        
        # For debugging
        self.debugInfo(root_node, num_simulations, time.time() - start_time)
        
        # Return the best action
        return root_node.best_action()
    
    def debugInfo(self, root_node, simulations, time_taken):
        """Print debug information about the search."""
        print(f"Ran {simulations} simulations in {time_taken:.3f} seconds")
        print("Available actions and their statistics:")
        for child in root_node.children:
            win_rate = child.total_reward / max(child.visits, 1)
            print(f"Action: {child.action}, Visits: {child.visits}, Win Rate: {win_rate:.3f}")


class MCTSNode:
    def __init__(self, gameState, agentIndex, parent=None, action=None):
        self.gameState = gameState
        self.agentIndex = agentIndex
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0
        # Initialize untried actions - important for proper expansion
        self.untried_actions = list(gameState.getLegalActions(agentIndex))
        
        # Remove STOP from consideration unless it's the only option
        if len(self.untried_actions) > 1 and Directions.STOP in self.untried_actions:
            self.untried_actions.remove(Directions.STOP)
    
    def is_fully_expanded(self):
        """Check if all possible actions have been tried from this node."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self):
        """Check if this node represents a terminal state."""
        return self.gameState.isOver()
    
    def select_child(self, exploration_weight):
        """
        Use UCB1 formula to select a child node, balancing
        exploitation with exploration.
        """
        # Ensure all children have been visited at least once
        unvisited = [child for child in self.children if child.visits == 0]
        if unvisited:
            return random.choice(unvisited)
        
        # Calculate UCB1 for all children
        log_parent_visits = math.log(self.visits)
        return max(
            self.children,
            key=lambda child: child.total_reward / child.visits + 
                exploration_weight * math.sqrt(log_parent_visits / child.visits)
        )
    
    def expand(self):
        """
        Add a new child node for a previously untried action and return it.
        """
        if not self.untried_actions:  # Safety check
            return self
            
        # Choose a random untried action
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        
        # Create a new child node
        next_state = self.gameState.generateSuccessor(self.agentIndex, action)
        child_node = MCTSNode(next_state, self.agentIndex, self, action)
        self.children.append(child_node)
        
        return child_node
    
    def rollout(self, max_depth=15):
        """
        Simulate a random game from the current state until terminal state
        or max_depth is reached. Uses epsilon-greedy strategy.
        """
        state = self.gameState
        depth = 0
        epsilon = 0.7  # Balance between greedy and random moves
        
        while depth < max_depth:
            # Get legal actions
            actions = state.getLegalActions(self.agentIndex)
            
            # Remove STOP if there are other options
            if len(actions) > 1 and Directions.STOP in actions:
                actions.remove(Directions.STOP)
                
            if not actions:
                break
                
            # Choose action: epsilon chance of greedy, (1-epsilon) chance of random
            if random.random() < epsilon:
                # Try different heuristics in rollout
                scores = [self._evaluate_action(state, a) for a in actions]
                max_score = max(scores)
                best_actions = [a for i, a in enumerate(actions) if scores[i] == max_score]
                action = random.choice(best_actions)  # Break ties randomly
            else:
                action = random.choice(actions)
            
            # Apply action
            state = state.generateSuccessor(self.agentIndex, action)
            depth += 1
            
            # Check if game is over
            if state.isOver():
                break
        
        # Return game score as reward
        return state.getScore() if self.agentIndex % 2 == 0 else -state.getScore()
    
    def _evaluate_action(self, state, action):
        """
        Evaluate an action using a combination of heuristics.
        This helps guide the rollout towards more promising states.
        """
        next_state = state.generateSuccessor(self.agentIndex, action)
        base_score = next_state.getScore() if self.agentIndex % 2 == 0 else -next_state.getScore()
        
        # Additional heuristics can be added here
        # For example, distance to food, distance to enemies, etc.
        
        return base_score

    def backpropagate(self, reward):
        """
        Update statistics for this node and all ancestors.
        """
        self.visits += 1
        self.total_reward += reward
        
        # Propagate up the tree
        if self.parent:
            self.parent.backpropagate(reward)
    
    def best_action(self):
        """
        Return the best action based on highest average reward.
        """
        if not self.children:
            legal_actions = self.gameState.getLegalActions(self.agentIndex)
            return random.choice(legal_actions) if legal_actions else Directions.STOP
        
        # Use visits as the primary criterion for the best action
        best_child = max(self.children, key=lambda child: child.visits)
        return best_child.action


def createTeam(firstIndex, secondIndex, isRed,
               first='MCTSAgent', second='MCTSAgent'):
    """
    Returns a team of MCTS agents.
    """
    return [MCTSAgent(firstIndex), MCTSAgent(secondIndex)]