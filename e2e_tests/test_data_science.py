"""E2E tests for data science packages — require uv + network."""

import shutil
import tempfile
from pathlib import Path

import pytest

from mcp_python_sandbox.executor import execute
from mcp_python_sandbox.script import build_script

pytestmark = pytest.mark.skipif(
    shutil.which("uv") is None,
    reason="uv not installed",
)

_TIMEOUT = 120


async def _run(script: str, deps: list[str] | None = None):
    """Helper: build script, write to tempfile, execute, return result."""
    final = build_script(
        script,
        extra_dependencies=deps,
        python_version="3.13",
    )
    with tempfile.TemporaryDirectory(prefix="mcp-e2e-") as tmpdir:
        path = Path(tmpdir) / "script.py"
        path.write_text(final, encoding="utf-8")
        return await execute(
            script_path=path,
            python_version="3.13",
            timeout=_TIMEOUT,
            sandbox=None,
            max_output_bytes=102_400,
        )


class TestPandas:
    @pytest.mark.asyncio
    async def test_dataframe_operations(self):
        """Create a DataFrame, groupby, describe."""
        result = await _run(
            """\
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age": [25, 30, 35, 28, 32],
    "salary": [50000, 60000, 75000, 55000, 80000],
    "department": ["Eng", "Eng", "Sales", "HR", "Sales"],
})

print("=== DataFrame ===")
print(df)

print("\\n=== Group by department ===")
print(df.groupby("department").agg({"salary": ["mean", "max"], "age": "mean"}))

print("\\n=== Describe ===")
print(df.describe())
""",
            deps=["pandas", "numpy"],
        )
        assert result.exit_code == 0
        assert "Alice" in result.stdout
        assert "Eng" in result.stdout
        assert "mean" in result.stdout

    @pytest.mark.asyncio
    async def test_data_pipeline(self):
        """Simulate a sales dataset with pivot tables and aggregations."""
        result = await _run(
            """\
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=n, freq="h"),
    "product": np.random.choice(["Widget A", "Widget B", "Gadget X", "Gadget Y"], n),
    "region": np.random.choice(["North", "South", "East", "West"], n),
    "quantity": np.random.poisson(10, n),
    "unit_price": np.random.uniform(5, 50, n).round(2),
})
df["revenue"] = df["quantity"] * df["unit_price"]

top = df.groupby("product")["revenue"].sum().sort_values(ascending=False)
print("=== Top Products ===")
for prod, rev in top.items():
    print(f"  {prod}: {rev:.2f}")

pivot = df.pivot_table(values="revenue", index="product", columns="region", aggfunc="mean").round(2)
print("\\n=== Pivot ===")
print(pivot.to_string())
""",
            deps=["pandas", "numpy"],
        )
        assert result.exit_code == 0
        assert "Widget" in result.stdout
        assert "Pivot" in result.stdout


class TestNumpy:
    @pytest.mark.asyncio
    async def test_linear_algebra(self):
        """Solve linear system, eigenvalues, SVD."""
        result = await _run(
            """\
import numpy as np

A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
b = np.array([8, -11, -3])
x = np.linalg.solve(A, b)
print(f"Solution: x={x[0]:.1f}, y={x[1]:.1f}, z={x[2]:.1f}")

M = np.array([[4, -2], [1, 1]])
eigenvalues, _ = np.linalg.eig(M)
print(f"Eigenvalues: {eigenvalues}")

data = np.random.RandomState(42).randn(5, 3)
U, S, Vt = np.linalg.svd(data, full_matrices=False)
err = np.linalg.norm(data - U @ np.diag(S) @ Vt)
print(f"SVD reconstruction error: {err:.2e}")
""",
            deps=["numpy"],
        )
        assert result.exit_code == 0
        assert "x=2.0" in result.stdout
        assert "y=3.0" in result.stdout
        assert "z=-1.0" in result.stdout


class TestScipy:
    @pytest.mark.asyncio
    async def test_statistics_and_optimization(self):
        """T-test, curve fitting, Pearson correlation."""
        result = await _run(
            """\
import numpy as np
from scipy import stats, optimize

np.random.seed(42)
group_a = np.random.normal(100, 10, 50)
group_b = np.random.normal(105, 10, 50)
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")
print(f"Significant: {p_value < 0.05}")

def model(x, a, b, c):
    return a * np.exp(-b * x) + c

xdata = np.linspace(0, 4, 50)
ydata = 2.5 * np.exp(-1.3 * xdata) + 0.5 + 0.2 * np.random.randn(50)
popt, _ = optimize.curve_fit(model, xdata, ydata)
print(f"Fitted: a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}")

x = np.random.randn(100)
y = 0.7 * x + 0.3 * np.random.randn(100)
r, p = stats.pearsonr(x, y)
print(f"Pearson r={r:.4f}, p={p:.2e}")
""",
            deps=["numpy", "scipy"],
        )
        assert result.exit_code == 0
        assert "Significant: True" in result.stdout
        assert "Pearson" in result.stdout

    @pytest.mark.asyncio
    async def test_time_series_analysis(self):
        """Trend detection and spectral analysis on synthetic time series."""
        result = await _run(
            """\
import pandas as pd
import numpy as np
from scipy import signal, stats

np.random.seed(42)
n = 365
dates = pd.date_range("2024-01-01", periods=n, freq="D")
trend = np.linspace(100, 150, n)
seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
noise = np.random.normal(0, 5, n)
values = trend + seasonal + noise

ts = pd.Series(values, index=dates, name="sales")

print(f"Mean: {ts.mean():.2f}, Std: {ts.std():.2f}")
print(f"Min: {ts.min():.2f} ({ts.idxmin().date()})")
print(f"Max: {ts.max():.2f} ({ts.idxmax().date()})")

monthly = ts.resample("ME").agg(["mean", "std"])
print(f"Monthly rows: {len(monthly)}")

slope, intercept, r_value, p_value, std_err = stats.linregress(range(n), values)
print(f"Trend slope: {slope:.4f}/day")
print(f"R-squared: {r_value**2:.4f}")

freqs, psd = signal.welch(values - np.mean(values), fs=1.0, nperseg=128)
dominant_freq = freqs[np.argmax(psd)]
dominant_period = 1 / dominant_freq if dominant_freq > 0 else float("inf")
print(f"Dominant period: {dominant_period:.1f} days")
""",
            deps=["pandas", "numpy", "scipy"],
        )
        assert result.exit_code == 0
        assert "Monthly rows: 12" in result.stdout
        assert "Trend slope:" in result.stdout


class TestScikitLearn:
    @pytest.mark.asyncio
    async def test_classification_pipeline(self):
        """RandomForest pipeline with cross-validation."""
        result = await _run(
            """\
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X, y = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_redundant=2, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=50, random_state=42)),
])

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
""",
            deps=["numpy", "scikit-learn"],
        )
        assert result.exit_code == 0
        assert "CV Accuracy:" in result.stdout
        assert "precision" in result.stdout
        # Accuracy should be reasonable
        assert "0.8" in result.stdout or "0.9" in result.stdout

    @pytest.mark.asyncio
    async def test_clustering_and_pca(self):
        """KMeans clustering + PCA dimensionality reduction."""
        result = await _run(
            """\
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=3, n_features=10, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

sil = silhouette_score(X, y_pred)
ari = adjusted_rand_score(y_true, y_pred)
print(f"Silhouette: {sil:.4f}")
print(f"ARI: {ari:.4f}")
print(f"Cluster sizes: {np.bincount(y_pred).tolist()}")

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
print(f"Explained variance: {pca.explained_variance_ratio_.round(4).tolist()}")
print(f"Total (3 PCs): {sum(pca.explained_variance_ratio_):.4f}")
""",
            deps=["numpy", "scikit-learn"],
        )
        assert result.exit_code == 0
        assert "ARI: 1.0" in result.stdout
        assert "Cluster sizes:" in result.stdout

    @pytest.mark.asyncio
    async def test_regression_with_pydantic_output(self):
        """Linear regression on synthetic housing data, output via Pydantic."""
        result = await _run(
            """\
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from pydantic import BaseModel

class ModelResult(BaseModel):
    r2: float
    mae: float
    coefficients: list[float]
    intercept: float
    feature_names: list[str]

np.random.seed(42)
n = 200
sqft = np.random.uniform(800, 3000, n)
bedrooms = np.random.randint(1, 6, n)
age = np.random.uniform(0, 50, n)
dist = np.random.uniform(1, 30, n)

price = 150*sqft + 20000*bedrooms - 1000*age - 2000*dist + 50000 + np.random.normal(0, 20000, n)

df = pd.DataFrame({"sqft": sqft, "bedrooms": bedrooms, "age": age, "distance": dist, "price": price})
features = ["sqft", "bedrooms", "age", "distance"]
X, y = df[features].values, df["price"].values

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

result = ModelResult(
    r2=round(r2_score(y, y_pred), 4),
    mae=round(mean_absolute_error(y, y_pred), 2),
    coefficients=[round(c, 2) for c in model.coef_],
    intercept=round(model.intercept_, 2),
    feature_names=features,
)
print(result.model_dump_json(indent=2))
""",
            deps=["numpy", "pandas", "scikit-learn", "pydantic>=2.0"],
        )
        assert result.exit_code == 0
        assert '"r2"' in result.stdout
        assert '"coefficients"' in result.stdout
        assert '"sqft"' in result.stdout


class TestMatplotlib:
    @pytest.mark.asyncio
    async def test_chart_generation(self):
        """Generate a 2x2 chart grid, save to PNG, report size."""
        result = await _run(
            """\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import io

np.random.seed(42)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), label="sin(x)")
axes[0, 0].plot(x, np.cos(x), label="cos(x)")
axes[0, 0].set_title("Trig")
axes[0, 0].legend()

axes[0, 1].hist(np.random.normal(0, 1, 1000), bins=30, edgecolor="black", alpha=0.7)
axes[0, 1].set_title("Histogram")

xs = np.random.randn(100)
ys = 2 * xs + np.random.randn(100) * 0.5
axes[1, 0].scatter(xs, ys, alpha=0.6, c=ys, cmap="viridis")
axes[1, 0].set_title("Scatter")

axes[1, 1].bar(["A", "B", "C", "D"], np.random.randint(10, 100, 4))
axes[1, 1].set_title("Bar")

plt.tight_layout()

buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=100)
size = buf.tell()
print(f"Charts: 4")
print(f"Image size: {size} bytes")
print(f"OK: {size > 1000}")
plt.close()
""",
            deps=["matplotlib", "numpy"],
        )
        assert result.exit_code == 0
        assert "Charts: 4" in result.stdout
        assert "OK: True" in result.stdout


class TestPolars:
    @pytest.mark.asyncio
    async def test_lazy_evaluation(self):
        """Polars lazy pipeline with groupby and window functions."""
        result = await _run(
            """\
import polars as pl
import random

random.seed(42)

dates = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 12, 31), eager=True)
n = len(dates)
df = pl.DataFrame({
    "date": dates,
    "category": [random.choice(["A", "B", "C"]) for _ in range(n)],
    "value": [random.gauss(100, 25) for _ in range(n)],
    "volume": [random.randint(10, 500) for _ in range(n)],
})

result = (
    df.lazy()
    .with_columns([
        pl.col("date").dt.month().alias("month"),
        (pl.col("value") * pl.col("volume")).alias("total_value"),
    ])
    .group_by(["month", "category"])
    .agg([
        pl.col("total_value").sum().alias("total"),
        pl.col("value").mean().alias("avg_value"),
    ])
    .sort(["month", "category"])
    .collect()
)

print(f"Result shape: {result.shape}")
print(f"Columns: {result.columns}")
print(f"Months covered: {result['month'].unique().sort().to_list()}")
print(f"Categories: {sorted(result['category'].unique().to_list())}")

ranked = (
    df.lazy()
    .with_columns([
        pl.col("value")
        .rank(method="ordinal", descending=True)
        .over("category")
        .alias("rank"),
    ])
    .filter(pl.col("rank") <= 3)
    .sort(["category", "rank"])
    .collect()
)
print(f"Top-3 per category: {ranked.shape[0]} rows")
""",
            deps=["polars"],
        )
        assert result.exit_code == 0
        assert "Result shape:" in result.stdout
        assert "['A', 'B', 'C']" in result.stdout
        assert "Top-3 per category: 9 rows" in result.stdout


class TestBeautifulSoup:
    @pytest.mark.asyncio
    async def test_html_parsing(self):
        """Parse an HTML table and compute stats."""
        result = await _run(
            """\
from bs4 import BeautifulSoup

html = '''
<html><body>
  <table id="data">
    <thead><tr><th>Name</th><th>Score</th><th>Grade</th></tr></thead>
    <tbody>
      <tr><td>Alice</td><td>95</td><td>A</td></tr>
      <tr><td>Bob</td><td>87</td><td>B+</td></tr>
      <tr><td>Charlie</td><td>92</td><td>A-</td></tr>
      <tr><td>Diana</td><td>78</td><td>C+</td></tr>
      <tr><td>Eve</td><td>98</td><td>A+</td></tr>
    </tbody>
  </table>
</body></html>
'''

soup = BeautifulSoup(html, "html.parser")
rows = soup.find("table", id="data").find_all("tr")[1:]

records = []
for row in rows:
    cols = [td.text for td in row.find_all("td")]
    records.append({"name": cols[0], "score": int(cols[1]), "grade": cols[2]})

avg = sum(r["score"] for r in records) / len(records)
top = max(records, key=lambda r: r["score"])
print(f"Parsed: {len(records)} rows")
print(f"Average: {avg:.1f}")
print(f"Top: {top['name']} ({top['score']})")
""",
            deps=["beautifulsoup4"],
        )
        assert result.exit_code == 0
        assert "Parsed: 5 rows" in result.stdout
        assert "Average: 90.0" in result.stdout
        assert "Top: Eve (98)" in result.stdout


class TestSympy:
    @pytest.mark.asyncio
    async def test_symbolic_math(self):
        """Solve equations, differentiate, integrate, matrix ops."""
        result = await _run(
            """\
from sympy import *

x, y = symbols("x y")

solutions = solve([x + y - 5, x - y - 1], [x, y])
print(f"System: {solutions}")

f = sin(x) * exp(-x)
print(f"diff: {diff(f, x)}")
print(f"integrate: {integrate(f, x)}")
print(f"limit: {limit(f, x, oo)}")

M = Matrix([[1, 2], [3, 4]])
print(f"det: {M.det()}")
print(f"inv: {M.inv()}")
""",
            deps=["sympy"],
        )
        assert result.exit_code == 0
        assert "{x: 3, y: 2}" in result.stdout
        assert "det: -2" in result.stdout
        assert "limit: 0" in result.stdout


class TestPEP723InlineMetadata:
    @pytest.mark.asyncio
    async def test_httpx_pydantic_inline(self):
        """Script with inline PEP 723 block declaring httpx + pydantic."""
        script = '''\
# /// script
# dependencies = ["httpx", "pydantic>=2.0"]
# requires-python = ">=3.11"
# ///

import httpx
from pydantic import BaseModel

class IPInfo(BaseModel):
    origin: str

resp = httpx.get("https://httpbin.org/ip")
info = IPInfo.model_validate(resp.json())
print(f"status: {resp.status_code}")
print(f"ip: {info.origin}")
'''
        with tempfile.TemporaryDirectory(prefix="mcp-e2e-") as tmpdir:
            path = Path(tmpdir) / "script.py"
            path.write_text(script, encoding="utf-8")
            result = await execute(
                script_path=path,
                python_version="3.13",
                timeout=_TIMEOUT,
                sandbox=None,
                max_output_bytes=102_400,
            )
        assert result.exit_code == 0
        assert "status: 200" in result.stdout
        assert "ip:" in result.stdout
