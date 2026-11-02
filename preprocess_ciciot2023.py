"""
CICIoT2023 preprocessing pipeline (PySpark)

Steps: load CSV(s) -> sample -> inspect labels -> binary target -> clean
       -> train/val/test split -> median imputation (fit on TRAIN only) -> save

Run:
    pip install pyspark
    python preprocess_ciciot2023.py
"""
import glob
import os

from pyspark.ml.feature import Imputer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# ----------------------------- CONFIG ---------------------------------
DATA_DIR = os.path.expanduser("~/Documents/DTS_301/dataset_directory")
# Reads every *.csv in the folder. Put BenignTraffic.pcap.csv in here too (see note).
CSV_GLOB = os.path.join(DATA_DIR, "*.csv")
OUT_DIR = os.path.expanduser("~/Documents/DTS_301/cleaned_output")

SAMPLE_SIZE = 50_000   # raise to 100_000 / 200_000 if Spark runs smoothly
SEED = 42              # agree with the group by Saturday 10 PM
TRAIN, VAL, TEST = 0.70, 0.15, 0.15
LABEL_COL = "Label"    # CICIoT2023 label column name
# ----------------------------------------------------------------------

spark = (
    SparkSession.builder.master("local[*]")
    .appName("CICIoT2023-preprocessing")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.sql.constraintPropagation.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

files = sorted(glob.glob(CSV_GLOB))
assert files, f"No CSV files found at {CSV_GLOB}"
print("Files found:", [os.path.basename(f) for f in files])

# 2. Load + sample ------------------------------------------------------
# The per-attack CSVs have no label column: the label is the file name
# (e.g. Backdoor_Malware.pcap.csv -> "Backdoor_Malware"). Add it per file.
parts = []
for f in files:
    label = os.path.basename(f).replace(".pcap.csv", "").replace(".csv", "")
    part = spark.read.csv(f, header=True, inferSchema=False)
    part = part.toDF(*[c.strip() for c in part.columns])
    if LABEL_COL in part.columns:      # some files may already carry a label
        part = part.withColumnRenamed(LABEL_COL, "Label")
    else:
        part = part.withColumn("Label", F.lit(label))
    parts.append(part)
df = parts[0]
for p in parts[1:]:
    df = df.unionByName(p, allowMissingColumns=True)

# Stratified sampling by label: split SAMPLE_SIZE evenly across labels; a label
# with fewer rows than its share contributes all of its rows and the unused
# quota is shared among the larger labels. This avoids a ~99% benign sample.
counts = {r["Label"]: r["count"] for r in df.groupBy("Label").count().collect()}
total = sum(counts.values())
remaining, quota = int(SAMPLE_SIZE * 1.03), {}   # 3% buffer: cleaning removes some rows
for i, (lab, n) in enumerate(sorted(counts.items(), key=lambda kv: kv[1])):
    take = min(n, remaining // (len(counts) - i))
    quota[lab] = take
    remaining -= take
fractions = {lab: min(1.0, quota[lab] / counts[lab]) for lab in counts}
print("Rows available per label:", counts)
print("Target rows per label:   ", quota)
df = df.sampleBy("Label", fractions, seed=SEED).cache()
print(f"Total rows available: {total:,} | sampled: {df.count():,}")

# 3. Check labels -------------------------------------------------------
print("\nLabel values and counts:")
df.groupBy("Label").count().orderBy(F.desc("count")).show(50, truncate=False)

# 4. Binary target (Benign = 0, any attack = 1) --------------------------
label_norm = F.lower(F.trim(F.col("Label")))
df = df.withColumn("target", F.when(label_norm.contains("benign"), 0).otherwise(1))
df = df.filter(F.col("Label").isNotNull())
print("Binary target distribution:")
df.groupBy("target").count().show()
if df.select("target").distinct().count() < 2:
    print("WARNING: only one class present. Add BenignTraffic.pcap.csv to the "
          "folder, otherwise the binary classifier has nothing to learn.")

# 5. Clean --------------------------------------------------------------
feature_cols = [c for c in df.columns if c not in ("Label", "target")]
# Cast to double and turn +/-inf and NaN into null (treated as missing).
# Done in ONE select: chaining withColumn in a loop builds a huge query plan
# that runs the Spark planner out of memory.
def _clean(c):
    x = F.col(f"`{c}`").cast(DoubleType())
    return (
        F.when(x.isin(float("inf"), float("-inf")) | F.isnan(x), None)
        .otherwise(x)
        .alias(c)
    )

df = df.select(*[_clean(c) for c in feature_cols], "Label", "target")

before = df.count()
df = df.dropDuplicates()
df = df.dropna(how="all", subset=feature_cols)        # rows with no usable features
print(f"Rows before cleaning: {before:,} | after: {df.count():,}")

# Trim to EXACTLY SAMPLE_SIZE rows after cleaning, keeping the same
# even-per-label rule (small labels keep everything, benign fills the rest).
from pyspark.sql.window import Window

# Cut the query plan: write the cleaned rows to disk and read them back, so
# later steps start from a tiny plan instead of the whole cleaning history.
tmp_clean = os.path.join(OUT_DIR, "_clean_tmp")
df.write.parquet(tmp_clean, mode="overwrite")
df = spark.read.parquet(tmp_clean)

orig_cols = df.columns
cc = {r["Label"]: r["count"] for r in df.groupBy("Label").count().collect()}
rem, q = SAMPLE_SIZE, {}
for i, (lab, n) in enumerate(sorted(cc.items(), key=lambda kv: kv[1])):
    take = min(n, rem // (len(cc) - i))
    q[lab] = take
    rem -= take
quota_df = spark.createDataFrame([(k, int(v)) for k, v in q.items()], ["Label", "_q"])
w = Window.partitionBy("Label").orderBy(F.rand(SEED))
df = (
    df.join(quota_df, "Label")
    .withColumn("_rn", F.row_number().over(w))
    .filter(F.col("_rn") <= F.col("_q"))
    .select(*orig_cols)
    .cache()
)
print(f"Final sample size after trimming: {df.count():,}")

# drop columns that are entirely null (Imputer cannot handle them)
non_null = df.select([F.count(c).alias(c) for c in feature_cols]).first().asDict()
feature_cols = [c for c in feature_cols if non_null[c] > 0]

# 6. Split FIRST, then impute (no leakage) -------------------------------
df = df.cache()
df.count()  # materialise once so the split is stable and not recomputed
train, val, test = df.randomSplit([TRAIN, VAL, TEST], seed=SEED)
train, val, test = train.cache(), val.cache(), test.cache()
print(f"Split sizes -> train: {train.count():,} | val: {val.count():,} | test: {test.count():,}", flush=True)
print("Fitting median imputer on train ...", flush=True)

imputer = Imputer(
    strategy="median",
    inputCols=feature_cols,
    outputCols=feature_cols,   # overwrite in place
)
model = imputer.fit(train)     # medians learned from TRAIN only
print("Imputer fitted. Applying to train/val/test ...", flush=True)
train, val, test = model.transform(train), model.transform(val), model.transform(test)

# 8. Save for Members 2 and 3 ---------------------------------------------
import shutil

os.makedirs(OUT_DIR, exist_ok=True)
for name, part in (("train", train), ("val", val), ("test", test)):
    tmp = os.path.join(OUT_DIR, f"_{name}_tmp")
    part.coalesce(1).write.csv(tmp, header=True, mode="overwrite")
    out = os.path.join(OUT_DIR, f"{name}.csv")
    shutil.move(glob.glob(os.path.join(tmp, "part-*.csv"))[0], out)
    shutil.rmtree(tmp)
    print("Saved", out)

shutil.rmtree(os.path.join(OUT_DIR, "_clean_tmp"), ignore_errors=True)
spark.stop()
