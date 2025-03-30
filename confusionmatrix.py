# confusion matrix for both wins and draws

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("final2.csv")

agents = sorted(set(df["RedAgent"]).union(set(df["BlueAgent"])))


win_matrix = pd.DataFrame(0, index=agents, columns=agents, dtype=float)
draw_matrix = pd.DataFrame(0, index=agents, columns=agents, dtype=float)


for _, row in df.iterrows():
    winner = row["Winner"]
    red = row["RedAgent"]
    blue = row["BlueAgent"]
    
    # Check for a draw:
    if winner == "DRAW":
        
        draw_matrix.loc[red, blue] += 1
        draw_matrix.loc[blue, red] += 1
    
    elif winner in agents:
        if winner == red:
            loser = blue
        elif winner == blue:
            loser = red
        else:
            continue 
        win_matrix.loc[winner, loser] += 1


np.fill_diagonal(win_matrix.values, np.nan)
np.fill_diagonal(draw_matrix.values, np.nan)


plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.heatmap(win_matrix, annot=True, fmt=".0f", cmap="Reds", linewidths=0.5, cbar_kws={'label': ' Number of Wins'})
plt.title("Confusion Matrix")
plt.xlabel("Opponent ")
plt.ylabel("Agent (Wins)")

plt.subplot(1, 2, 2)
sns.heatmap(draw_matrix, annot=True, fmt=".0f", cmap="Purples", linewidths=0.5, cbar_kws={'label': 'Number of Draws'})
plt.title("Confusion Matrix")
plt.xlabel("Opponent")
plt.ylabel("Agent(Draws)")

plt.tight_layout()
plt.show()


win_matrix.to_csv("win_confusion_matrix.csv")
draw_matrix.to_csv("draw_confusion_matrix.csv")