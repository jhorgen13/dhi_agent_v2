from pathlib import Path
import pandas as pd

def get_topic_folders(base_path="data/Numeric Tables"):
    base_dir = Path(base_path)
    return sorted([f for f in base_dir.iterdir() if f.is_dir()])

def load_csv_topic_data(topic_path):
    csv_files = list(Path(topic_path).glob("*.csv"))
    all_dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df["__source_file__"] = file.name
            all_dataframes.append(df.dropna(how="all").dropna(axis=1, how="all"))
        except Exception as e:
            print(f"⚠️ Skipping {file.name}: {e}")
    return pd.concat(all_dataframes, ignore_index=True) if all_dataframes else pd.DataFrame()
