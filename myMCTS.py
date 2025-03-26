from captureAgents import CaptureAgent
import random, time, util, math, os, json
from game import Directions
import game

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH) as f:
    MCTS_CONFIG = json.load(f)

def createTeam(firstIndex, secondIndex, isRed,
               first='ActsAgent', second='ActsAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class Node:
    def __init__(self, gameState, agentIndex, parent=None, action=None):
        self.gameState = gameState
        self.agentIndex = agentIndex
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.totalReward = 0.0
        self.untriedActions = gameState.getLegalActions(agentIndex)
        if Directions.STOP in self.untriedActions:
            self.untriedActions.remove(Directions.STOP)

class ActsAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.stats = {
            "totalDecisionTime": 0.0,
            "actionsTaken": 0
        }

    def chooseAction(self, gameState):
        start = time.time()
        root = Node(gameState, self.index)
        time_limit = 0.2
        while time.time() - start < time_limit:
            node = self.select(root)
            if node.untriedActions:
                node = self.expand(node)
            reward = self.simulate(node)
            self.backpropagate(node, reward)

        action = self.bestChild(root, Cp=0).action

        # Track decision time
        end = time.time()
        self.stats["totalDecisionTime"] += (end - start)
        self.stats["actionsTaken"] += 1

        return action

    def select(self, node):
        Cp = MCTS_CONFIG.get("explorationConstant", 1 / math.sqrt(2))
        while not node.untriedActions and node.children:
            node = self.bestChild(node, Cp)
        return node

    def expand(self, node):
        action = node.untriedActions.pop()
        nextState = node.gameState.generateSuccessor(self.index, action)
        child = Node(nextState, self.index, parent=node, action=action)
        node.children.append(child)
        return child

    def simulate(self, node):
        state = node.gameState
        depth = MCTS_CONFIG.get("rolloutDepth", 5)
        epsilon = MCTS_CONFIG.get("epsilon", 0.2)
        useHeuristic = MCTS_CONFIG.get("useHeuristicRollouts", False)

        for _ in range(depth):
            actions = state.getLegalActions(self.index)
            actions = [a for a in actions if a != Directions.STOP]
            if not actions:
                break

            if useHeuristic:
                if random.random() < epsilon:
                    action = random.choice(actions)
                else:
                    bestScore = float('-inf')
                    bestAction = None
                    for a in actions:
                        successor = state.generateSuccessor(self.index, a)
                        score = self.evaluate(successor)
                        if score > bestScore:
                            bestScore = score
                            bestAction = a
                    action = bestAction
            else:
                action = random.choice(actions)

            state = state.generateSuccessor(self.index, action)

        return self.evaluate(state)

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.totalReward += reward
            node = node.parent

    def bestChild(self, node, Cp):
        bestScore = float('-inf')
        bestChild = None
        for child in node.children:
            if child.visits == 0:
                continue
            exploit = child.totalReward / child.visits
            explore = Cp * math.sqrt(2 * math.log(node.visits) / child.visits)
            score = exploit + explore
            if score > bestScore:
                bestScore = score
                bestChild = child
        return bestChild

    def evaluate(self, gameState):
        features = util.Counter()
        state = gameState.getAgentState(self.index)
        pos = state.getPosition()
        foodList = self.getFood(gameState).asList()
        walls = gameState.getWalls()

        if foodList:
            minFoodDist = min([self.getMazeDistance(pos, food) for food in foodList])
            features['distanceToFood'] = -minFoodDist

        features['carrying'] = state.numCarrying

        enemyIndices = self.getOpponents(gameState)
        for enemy in enemyIndices:
            enemyState = gameState.getAgentState(enemy)
            if not enemyState.isPacman and enemyState.getPosition() is not None:
                dist = self.getMazeDistance(pos, enemyState.getPosition())
                if dist <= 5:
                    features['ghostProximity'] += 1.0 / (dist + 0.1)

        if state.numCarrying > 0:
            midX = (gameState.data.layout.width - 2) // 2
            homeX = midX if self.red else midX + 1
            homePositions = [(homeX, y) for y in range(gameState.data.layout.height)
                             if not walls[homeX][y]]
            minHomeDist = min([self.getMazeDistance(pos, hp) for hp in homePositions])
            features['distanceToHome'] = -minHomeDist

        weights = {
            'distanceToFood': 1.5,
            'carrying': 10.0,
            'ghostProximity': -8.0,
            'distanceToHome': 2.0,
        }

        return features * weights

    def final(self, gameState):
        CaptureAgent.final(self, gameState)

        teamScore = gameState.getScore()
        won = teamScore > 0 if self.red else teamScore < 0

        avgTime = self.stats["totalDecisionTime"] / max(1, self.stats["actionsTaken"])

        result = {
            "agentType": "Guided" if MCTS_CONFIG.get("useHeuristicRollouts") else "Vanilla",
            "won": won,
            "score": teamScore,
            "avgDecisionTime": round(avgTime, 4),
            "epsilon": MCTS_CONFIG["epsilon"],
            "rolloutDepth": MCTS_CONFIG["rolloutDepth"],
            "Cp": MCTS_CONFIG["explorationConstant"]
        }

        log_path = os.path.join(os.path.dirname(__file__), "experiment_logs.csv")
        file_exists = os.path.isfile(log_path)
        with open(log_path, "a") as f:
            if not file_exists:
                f.write(",".join(result.keys()) + "\n")
            f.write(",".join(str(v) for v in result.values()) + "\n")
