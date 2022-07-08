import pandas as pd
import numpy as np
import plotly.express as px
from timm.models.registry import is_model, model_entrypoint

df1 = pd.read_csv("datasets/results-imagenet-real.csv")
print(df1.shape)

df2 = pd.read_csv("datasets/benchmark-infer-amp-nchw-pt110-cu113-rtx3090.csv")
print(df2.shape)

df = df1.merge(df2)
print(df.shape)
print(df.columns)

df["infer_gmacs"] = np.log1p(df["infer_gmacs"]) + 0.5
df["pareto"] = df.apply(
    lambda r: df[
        (df["top1"] > r["top1"])
        & (df["infer_samples_per_sec"] > r["infer_samples_per_sec"])
    ].shape[0],
    axis=1,
)
print(df["pareto"].describe())
df["architecture"] = df["model"].apply(
    lambda m: model_entrypoint(m).__module__.split(".")[2] if is_model(m) else "unknown"
)
df = df.sort_values("architecture")

fig = px.scatter(
    df[(df["pareto"] < 5) & (df["architecture"] != "unknown")],
    x="top1",
    y="infer_samples_per_sec",
    size="infer_gmacs",
    color="architecture",
    hover_name="model",
    log_y=True,
    # size_max=1500,
)
fig.write_html("datasets/timm_backbones.html")
