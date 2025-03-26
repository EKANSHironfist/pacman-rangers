import os
import json
import itertools
import subprocess
from time import sleep

# === Paths ===
CONFIG_PATH = "config.json"  # Used by your myTeam.py agent
LOG_PATH = "experiment_test.csv"
LAYOUT = "layouts/defaultCapture.lay"
CAPTURE_COMMAND_TEMPLATE = "python capture.py -r baselineTeam -b myTeam -l {layout} -q -n 1"

# === Hyperparameter Grid ===
rollout_depths = [5]
epsilons = [0.2]
exploration_constants = [0.707]
use_heuristics = [True]
games_per_config = 2

# === Clear log file if it exists ===
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)

# === Generate all hyperparameter combinations ===
combinations = list(itertools.product(rollout_depths, epsilons, exploration_constants, use_heuristics))

# === Run experiments ===
for i, (depth, eps, cp, heuristic) in enumerate(combinations):
    print(f"\n[{i+1}/{len(combinations)}] Testing config: rolloutDepth={depth}, epsilon={eps}, Cp={cp}, heuristic={heuristic}")

    # Write config.json for the MCTS agent to use
    config = {
        "rolloutDepth": depth,
        "epsilon": eps,
        "explorationConstant": cp,
        "useHeuristicRollouts": heuristic
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    # Run multiple games per config
    for g in range(games_per_config):
        print(f"  - Game {g + 1}/{games_per_config}", end="... ")
        try:
            cmd = CAPTURE_COMMAND_TEMPLATE.format(layout=LAYOUT)
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
            print("done.")
        except subprocess.CalledProcessError:
            print("❌ failed.")
            continue

        sleep(0.5)  # Optional short delay between games

print("\n✅ All experiments complete.")
print(f"Results stored in: {LOG_PATH}")
