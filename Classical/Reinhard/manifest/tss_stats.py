import pandas as pd

df = pd.read_csv("/homes/wdkarim/manifests/brca_matching_slide_tss_manifest.csv")

stats = (
    df.groupby("tss")
      .agg(
          n_slides=("slide_id", "count"),
          n_events=("event_os", "sum"),
          median_time=("time_os_months", "median")
      )
      .sort_values(["n_slides", "n_events"], ascending=False)
      .reset_index()
)

print(stats.to_string(index=False))
stats.to_csv("/homes/wdkarim/manifests/tss_stats.csv", index=False)
print("\nSaved: /homes/wdkarim/manifests/tss_stats.csv")
