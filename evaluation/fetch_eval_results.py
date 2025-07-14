from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import matplotlib.pyplot as plt
from langsmith import Client

# Initialize using .env (older client-compatible)
client = Client()

# Settings
experiment_name = "csv_agent_batch_eval"

print(f"🔎 Fetching runs for experiment: {experiment_name}")
runs = client.list_runs(execution_order=1, run_type="chain")

# Collect feedback records
records = []
for run in runs:
    if run.experiment_name != experiment_name:
        continue
    feedback = run.feedback or {}
    records.append({
        "run_id": run.id,
        "question": run.inputs.get("question", ""),
        "answer": run.outputs.get("answer", ""),
        "correct": feedback.get("correct"),
        "has_number": feedback.get("has_number"),
        "out_of_range": feedback.get("out_of_range"),
        "trace_url": f"https://smith.langchain.com/r/{run.id}"
    })

# Save results
df = pd.DataFrame(records)
df.to_csv("evaluation_results.csv", index=False)
print("✅ Results saved to evaluation_results.csv")

# Summary stats
summary = {
    "total": len(df),
    "correct": int(df["correct"].sum()),
    "has_number": int(df["has_number"].sum()),
    "passed_range_check": int(df["out_of_range"].sum())
}
print("📊 Summary:")
for k, v in summary.items():
    print(f"  {k}: {v}")

# Plot chart
df_plot = pd.DataFrame.from_dict(summary, orient="index", columns=["Count"])
df_plot.drop("total", errors="ignore").plot(kind="bar", legend=False)
plt.title("Evaluation Results Summary")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("evaluation_results_summary.png")
plt.show()
print("📈 Chart saved to evaluation_results_summary.png")
