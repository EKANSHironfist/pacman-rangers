import subprocess
import re
import os
import pandas as pd
import itertools
from datetime import datetime

# === CONFIGURATION ===
AGENTS = ["myMCTSV2", "myMCTSV2_sorting","myMCTSV2_weighted_rollout"]      # you can add your model pair here 
CAPTURE_SCRIPT = "capture.py"
OUTPUT_LOG = "tournament_results.csv"
MATCHES_PER_PAIR = 30 # change that number to 50 if you want to run 100 matches per  model pair
VISUAL = True           # Set to False to run silently
MATCH_TIMEOUT = 600          # Max seconds per match

results = []

def run_match(red, blue):
    command = [
        "python", CAPTURE_SCRIPT,
        "-r", f"models/{red}.py",
        "-b", f"models/{blue}.py"
    ]
    if not VISUAL:
        command.append("-q")  # Only add quiet mode if visualization is OFF

    try:
        print(f"▶️ {red} (Red) vs {blue} (Blue)")
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=MATCH_TIMEOUT, universal_newlines=True)

        # Try to read the score file written by capture.py
        try:
            with open("score", "r") as f:
                final_score = int(f.read().strip())
            if final_score > 0:
                red_score = final_score
                blue_score = 0
                winner = red
            elif final_score < 0:
                red_score = 0
                blue_score = abs(final_score)
                winner = blue
            else:
                red_score = blue_score = 0
                winner = "DRAW"
            print(f"\U0001F3C6 Winner: {winner}  Score: {red_score} to {blue_score}")
        except FileNotFoundError:
            red_score, blue_score, winner = "ERR", "ERR", "ERROR"
            print(" Score file not found.")

    except subprocess.TimeoutExpired:
        print(f" Match timeout!")
        red_score, blue_score, winner = "TIMEOUT", "TIMEOUT", "TIMEOUT"

    except subprocess.CalledProcessError:
        print(f" Match error!")
        red_score, blue_score, winner = "ERR", "ERR", "ERROR"

    return {
        "MatchID": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "RedAgent": red,
        "BlueAgent": blue,
        "RedScore": red_score,
        "BlueScore": blue_score,
        "Winner": winner
    }

# === RUN MATCHES ===
for red in ["myHeuristic", "myMCTS"]:
    for blue in AGENTS:
        for _ in range(MATCHES_PER_PAIR):
            result = pd.DataFrame([run_match(red, blue)])
            if not os.path.exists(OUTPUT_LOG):
                result.to_csv(OUTPUT_LOG)
            else:
                df = pd.read_csv(OUTPUT_LOG)
                df = pd.concat([df, result], ignore_index=True)
                df.to_csv(OUTPUT_LOG)

# === SAVE TO CSV ===
# df = pd.DataFrame(results)
# df.to_csv(OUTPUT_LOG, index=False)
# print(f"\n✅ Tournament complete. Results saved to: {OUTPUT_LOG}")
