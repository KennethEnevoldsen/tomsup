import sys

sys.path.append("/Users/au568658/Desktop/Academ/Projects/tomsup")
import tomsup as ts
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load data
experiment_df = pd.read_csv(
    "CogSci19_MatchingPennies_TOM_data.csv", dtype={"trial": np.int32}
)
# Make column for storing observer tom states
experiment_df["observer_tom_state"] = np.nan
# Get payoff matrix
penny_comp = ts.PayoffMatrix("penny_competitive")

# Subset the dataframe
for participant_id in tqdm(np.unique(experiment_df["participant.code"])):
    temp_df_3 = experiment_df[experiment_df["participant.code"] == participant_id]
    for level in np.unique(temp_df_3["tom_level"]):
        temp_df_2 = temp_df_3[temp_df_3["tom_level"] == level]
        for framing in np.unique(temp_df_2["Framing"]):
            temp_df_1 = temp_df_2[temp_df_2["Framing"] == framing]

            # initialize a k-ToM agent
            agent = ts.TOM(level=3, save_history=True)

            # Learn and act for next round
            choice_agent = agent.compete(p_matrix=penny_comp, agent=0, op_choice=None)
            # Force own choice at next round to be consistent with dataframe
            agent.choice = temp_df_1[
                temp_df_1["trial"] == min(np.unique(temp_df_1["trial"]))
            ].tom_decision.iloc[0]

            # Go through each of the trials
            for trial in np.unique(temp_df_1["trial"])[:]:
                # get the opponent's choice from the dataframe
                prev_choice_human = temp_df_1[
                    temp_df_1["trial"] == trial
                ].decision.iloc[0]

                # Learn and act for next round
                choice_agent = agent.compete(
                    p_matrix=penny_comp,
                    agent=temp_df_1[temp_df_1["trial"] == trial].tom_role.iloc[0],
                    op_choice=prev_choice_human,
                )

                # Force own choice at next round to be consistent with dataframe
                try:
                    agent.choice = temp_df_1[
                        temp_df_1["trial"] == (trial + 1)
                    ].tom_decision.iloc[0]
                # Pass on the last trial where the dataframe stops
                except:
                    pass

                # Add to dataframe
                experiment_df.loc[
                    (experiment_df["participant.code"] == participant_id)
                    & (experiment_df["tom_level"] == level)
                    & (experiment_df["Framing"] == framing)
                    & (experiment_df["trial"] == trial),
                    "observer_tom_state",
                ] = str(agent.get_internal_states())

# Save to CSV
experiment_df.to_csv("CogSci19_MatchingPennies_TOM_data_updated.csv", index=False)
