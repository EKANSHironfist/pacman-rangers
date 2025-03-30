import pandas as pd
import trueskill

# Load the combined tournament results CSV.

df = pd.read_csv("final2.csv")

# Initialize TrueSkill environment.
env = trueskill.TrueSkill(mu=25.0, sigma=8.333, beta=4.167, tau=0.0833, draw_probability=0.0) # these are the default values used in the research paper recommended in the assignmnet pdf

# Extract agents.
agents = pd.unique(df[['RedAgent', 'BlueAgent']].values.ravel())


ratings = {agent: env.create_rating() for agent in agents}


win_counts = {agent: 0 for agent in agents}
match_counts = {agent: 0 for agent in agents}

# Process each match and update TrueSkill ratings
for _, row in df.iterrows():
    red = row["RedAgent"]
    blue = row["BlueAgent"]
    winner = row["Winner"]

    # Update match counts for both agents.
    match_counts[red] += 1
    match_counts[blue] += 1

    # Skip matches with invalid outcomes.
    if winner not in [red, blue, "DRAW"]:
        continue

    # Retrieve current ratings.
    red_rating = ratings[red]
    blue_rating = ratings[blue]

    if winner == red:
        new_red, new_blue = env.rate_1vs1(red_rating, blue_rating)
        win_counts[red] += 1
    elif winner == blue:
        new_blue, new_red = env.rate_1vs1(blue_rating, red_rating)
        win_counts[blue] += 1
    elif winner == "DRAW":
        new_red, new_blue = env.rate_1vs1(red_rating, blue_rating, drawn=True)
    
    ratings[red] = new_red
    ratings[blue] = new_blue

# Calculate win percentages for each agent.
win_percentages = {
    agent: (win_counts[agent] / match_counts[agent] * 100) if match_counts[agent] > 0 else 0
    for agent in agents
}

# Prepare the final ratings table.
final_ratings = []
for agent, rating in ratings.items():
    conservative = rating.mu - 3 * rating.sigma
    final_ratings.append({
        "Agent": agent,
        "Mu": round(rating.mu, 2),
        "Sigma": round(rating.sigma, 2),
        "TrueSkill Score (μ - 3σ)": round(conservative, 2),
        "Win Percentage": round(win_percentages[agent], 2)
    })

final_df = pd.DataFrame(final_ratings)
final_df = final_df.sort_values(by="TrueSkill Score (μ - 3σ)", ascending=False)

#  final TrueSkill ratings with win percentage to a CSV file.
final_df.to_csv("trueskill.csv", index=False)

print(final_df)