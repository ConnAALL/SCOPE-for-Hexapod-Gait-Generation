#!/usr/bin/env python3
"""
preprocess_csv_to_npy_parallel.py

Read all CSVs from a raw-data folder (including subfolders), compute per-generation best & average
for DistanceDifference, StabilityRatio, and TotalFitness in parallel,
and save each result as a .npy file in subfolders named after the CSV's parent folder.
Optionally extract top-K best individuals from the final generation across all files.
"""

import os
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool, cpu_count
import argparse
from collections import defaultdict

# Ensure spawn start method for compatibility across platforms
multiprocessing.set_start_method('spawn', force=True)

METRICS = ["DistanceDifference", "StabilityRatio", "TotalFitness"]

def find_csv_files(root_dir: str):
    csv_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.csv'):
                csv_paths.append(os.path.join(dirpath, fname))
    return sorted(csv_paths)

def process_file(csv_path: str, out_base: str):
    folder_name = os.path.basename(os.path.dirname(csv_path))
    out_dir = os.path.join(out_base, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(out_dir, f"{base}.npy")

    if os.path.exists(out_path):
        print(f"[{os.getpid()}] → skipping already processed: {folder_name}/{base}.npy")
        return

    df = pd.read_csv(csv_path)
    if "Generation" not in df.columns:
        print(f"Warning: 'Generation' column not found in {csv_path}, skipping.")
        return

    unique_gens = df["Generation"].nunique()
    if unique_gens <= 4999:
        print(f"[{os.getpid()}] → skipping {csv_path} (only {unique_gens} generations)")
        return

    if "TotalFitness" not in df.columns:
        df["TotalFitness"] = df["DistanceDifference"] * df["StabilityRatio"]

    gens = np.sort(df["Generation"].unique())
    result = {"Generation": gens}

    for m in METRICS:
        bests, avgs = [], []
        best_so_far = -np.inf
        for g in gens:
            vals = df.loc[df["Generation"] == g, m].to_numpy()
            if vals.size == 0:
                bests.append(best_so_far)
                avgs.append(np.nan)
            else:
                gen_best = vals.max()
                best_so_far = max(best_so_far, gen_best)
                bests.append(best_so_far)
                avgs.append(vals.mean())
        result[f"{m}_best"] = np.array(bests)
        result[f"{m}_avg"] = np.array(avgs)

    np.save(out_path, result)
    print(f"[{os.getpid()}] → saved {folder_name}/{base}.npy")

from collections import defaultdict

def extract_topk_best(csv_files, topk, out_base):
    all_individuals = []
    best_by_folder = defaultdict(lambda: {"fitness": -np.inf, "file": None})

    for path in csv_files:
        df = pd.read_csv(path)
        if "TotalFitness" not in df.columns:
            if "DistanceDifference" in df.columns and "StabilityRatio" in df.columns:
                df["TotalFitness"] = df["DistanceDifference"] * df["StabilityRatio"]
            else:
                continue  # skip if essential columns are missing

        if df.empty:
            continue

        all_individuals.append(df)

        folder = os.path.basename(os.path.dirname(path))
        local_best = df.loc[df["TotalFitness"].idxmax()]
        local_fitness = local_best["TotalFitness"]

        if local_fitness > best_by_folder[folder]["fitness"]:
            best_by_folder[folder]["fitness"] = local_fitness
            best_by_folder[folder]["file"] = path

    if not all_individuals:
        print("No valid data found for top-k extraction.")
        return

    combined = pd.concat(all_individuals, ignore_index=True)
    topk_df = combined.nlargest(topk, "TotalFitness")
    topk_arr = topk_df.to_records(index=False)
    np.save(os.path.join(out_base, "topk_best.npy"), topk_arr)
    print(f"Saved top-{topk} best individuals across all generations to topk_best.npy")

    print("\n🏆 Best CSV file per folder based on all-time TotalFitness:")
    for folder, info in sorted(best_by_folder.items()):
        print(f"📂 {folder}: {info['file']} (TotalFitness = {info['fitness']})")


def main(raw_dir: str, out_base: str, topk: int):
    os.makedirs(out_base, exist_ok=True)
    csv_files = find_csv_files(raw_dir)

    if not csv_files:
        print(f"No CSV files found in {raw_dir} or its subfolders")
        return

    source_folders = sorted({os.path.basename(os.path.dirname(p)) for p in csv_files})
    print(f"Detected CSV folders: {source_folders}")

    print(f"Starting processing with {cpu_count()} workers and {len(csv_files)} files...")
    tasks = [(path, out_base) for path in csv_files]

    with Pool(processes=cpu_count()) as pool:
        pool.starmap(process_file, tasks)

    if topk > 0:
        extract_topk_best(csv_files, topk, out_base)

    print("All files processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="./raw_data_input", help="Directory containing raw CSV files")
    parser.add_argument("--out_base", type=str, default="./processed_data", help="Output base directory")
    parser.add_argument("--topk", type=int, help="Top K best individuals to extract from last generations")

    args = parser.parse_args()
    main(args.raw_dir, args.out_base, args.topk)
