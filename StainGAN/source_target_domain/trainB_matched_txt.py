from pathlib import Path
import random

data_dir = Path("/homes/wdkarim/stain_norm/staingan_data/BH_vs_nonBH_256")
trainA = data_dir / "trainA.txt"
trainB = data_dir / "trainB.txt"
trainB_matched = data_dir / "trainB_matched.txt"

seed = 42

lines_A = [line.strip() for line in trainA.read_text().splitlines() if line.strip()]
lines_B = [line.strip() for line in trainB.read_text().splitlines() if line.strip()]

n_A = len(lines_A)
n_B = len(lines_B)

if n_B < n_A:
    raise SystemExit(f"ERROR: trainB has fewer lines than trainA ({n_B} < {n_A})")

rng = random.Random(seed)
subset_B = rng.sample(lines_B, n_A)

trainB_matched.write_text("\n".join(subset_B) + "\n", encoding="utf-8")

print(f"trainA lines:         {n_A:,}")
print(f"trainB original:      {n_B:,}")
print(f"trainB matched saved: {len(subset_B):,}")
print(f"output: {trainB_matched}")