from pathlib import Path

import pandas as pd

from stac import friedman_test, holm_test

# Define paths
RESULTS = Path("results")
FRIEDMAN = Path("friedman")

# Create directories
FRIEDMAN.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # load results

    tuning_scorings = [
        "f1",
    ]
    for ts in tuning_scorings:
        # load specific results
        print(f"\nCalculating friedman for {ts}...")
        df = pd.read_csv(RESULTS / f"{ts}_results.csv")

        # set-up folder
        TUNING_FRIEDMAN = FRIEDMAN / ts
        TUNING_FRIEDMAN.mkdir(parents=True, exist_ok=True)

        # define metrics
        metrics = ["prc", "recall", "mcc", "f1"]

        for metric in metrics:
            # isolate dataset metric and model
            friedman_df = df[["dataset", "model", metric]]

            # pivot table
            friedman_df = friedman_df.pivot(
                index="dataset", columns="model", values=metric
            )

            # make positive values negative
            friedman_df = -1 * friedman_df

            # compute the friedman test
            friedman_df = friedman_df.reset_index(drop=True)
            statistic, p_value, ranking, rank_cmp = friedman_test(
                *friedman_df.to_dict().values()
            )
            friedman = pd.DataFrame(index=friedman_df.columns.tolist())
            friedman["ranking"] = ranking
            friedman = friedman.sort_values(by="ranking")
            friedman.to_csv(TUNING_FRIEDMAN / f"{metric}_friedman.csv")
            print(f"Friedman for {metric} is finished!")

            # post-hock: create a dictionary with format 'groupname':'pivotal quantity'
            ranking_dict = dict(zip(friedman_df.columns, ranking))

            # post-hoc: holms test
            comparisons, z_values, p_values, adj_p_values = holm_test(
                ranking_dict, control="Decision Tree"
            )
            holm = pd.DataFrame(index=comparisons)
            holm["z_values"] = z_values
            holm["p_values"] = p_values
            holm["adj_p_values"] = adj_p_values
            holm.to_csv(TUNING_FRIEDMAN / f"{metric}_holm.csv")
            print(f"Holm for {metric} is finished!")
