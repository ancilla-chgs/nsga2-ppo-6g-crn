import os
import pandas as pd
import matplotlib.pyplot as plt

def run_batch_evaluations(dataset_list, results_dir="results"):
    print(f" Evaluating these datasets: {dataset_list}")
    summary_data = []

    for ds in dataset_list:
        print(f"Running: python evaluate.py --dataset {ds}")
        dataset_name = ds.split('.')[0]
        exit_code = os.system(f"python evaluate.py --dataset {ds}")

        if exit_code != 0:
            print(f" Evaluation failed for {ds} (Exit Code: {exit_code})")
            continue

        result_path = os.path.join(results_dir, f"final_results_{dataset_name}.csv")
        if os.path.exists(result_path):
            result = pd.read_csv(result_path)
            result["Dataset"] = ds
            summary_data.append(result)
        else:
            print(f" Result not found for {ds}")

    if summary_data:
        combined_df = pd.concat(summary_data, ignore_index=True)
        combined_df.to_csv("all_dataset_evaluation_summary.csv", index=False)
        print("\nCombined summary saved as 'all_dataset_evaluation_summary.csv'")

        # Plot Reward Comparison
        plt.figure(figsize=(12, 5))
        for dataset in combined_df["Dataset"].unique():
            subset = combined_df[combined_df["Dataset"] == dataset]
            plt.bar(subset["Algorithm"] + " (" + dataset.replace(".csv", "") + ")", subset["Average Reward"])

        plt.xticks(rotation=45)
        plt.title("Average Reward Comparison Across Datasets")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/reward_comparison_{dataset_name}.png")
        plt.close()

        # Plot Fairness Comparison
        plt.figure(figsize=(12, 5))
        for dataset in combined_df["Dataset"].unique():
            subset = combined_df[combined_df["Dataset"] == dataset]
            plt.bar(subset["Algorithm"] + " (" + dataset.replace(".csv", "") + ")", subset["Fairness (Jain Index)"])

        plt.xticks(rotation=45)
        plt.title("Fairness (Jain Index) Comparison Across Datasets")
        plt.ylabel("Fairness")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/fairness_comparison_{dataset_name}.png")
        plt.close()

        plt.show()

        return combined_df
    else:
        print(" No valid evaluation results found.")
        return None

if __name__ == "__main__":
    print("✅ Starting batch evaluation script")
    dataset_list = [
        "val.csv",
        "val_low_traffic.csv",
        "val_high_traffic.csv",
        "val_noisy.csv"
    ]
    run_batch_evaluations(dataset_list)
