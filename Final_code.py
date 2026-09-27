from pathlib import Path
import gc
import json
import ast
import re
import csv
import time
import unicodedata
import warnings
import zipfile

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
    HAVE_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_LIGHTGBM = False

from sklearn.model_selection import train_test_split

try:
    import psutil
except ImportError:
    psutil = None


ZIP_PATH = Path.home() / "Downloads" / "6ab10eb3b23ba_student_resource.zip"
OUTPUT_DIR = Path.home() / "amazon_ml_entity_resolution_output"

RANDOM_SEED = 42

BUCKET_CAP = 100
MAX_PAIRS_PER_KEY = 5_000
MAX_CANDIDATES_PER_S1 = 8

VALIDATION_ENTITY_FRACTION = 0.2
NEGATIVE_PER_POSITIVE = 8
FEATURE_BATCH = 50_000
MEMORY_SAFETY_FRACTION = 0.75

FEATURE_NAMES = [
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "address_ratio",
    "address_token_set",
    "name_compact_exact",
    "address_digits_exact",
    "country_match",
    "name_signature_exact",
    "name_length_similarity",
]


def banner(text):
    print("\n" + text)


def section(text):
    print("\n" + text)


def find_header_column(headers, candidates, required=True):
    exact = {str(c).lower(): c for c in headers}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    for col in headers:
        low = str(col).lower()
        for candidate in candidates:
            if candidate.lower() in low:
                return col
    if required:
        raise KeyError(f"Could not find one of {candidates}. Available columns: {headers}")
    return None


def choose_source_columns(headers):
    id_col = find_header_column(
        headers,
        ["id", "entity_id", "record_id", "source1_id", "source2_id", "source3_id",
         "source_1_id", "source_2_id", "source_3_id"],
        required=True,
    )
    name_col = find_header_column(
        headers,
        ["name", "company_name", "business_name", "entity_name", "merchant_name",
         "supplier_name", "title"],
        required=False,
    )
    address_col = find_header_column(
        headers,
        ["address", "full_address", "street", "location", "address_line", "address1", "addr"],
        required=False,
    )
    country_col = find_header_column(
        headers, ["country", "country_name", "nation"], required=False,
    )
    if name_col is None:
        alternatives = [c for c in headers if c != id_col]
        if not alternatives:
            raise KeyError("No usable name column found.")
        name_col = alternatives[0]
    return {"id": id_col, "name": name_col, "address": address_col, "country": country_col}


def choose_gt_columns(headers):
    return {
        "s1": find_header_column(
            headers, ["source1_id", "source_1_id", "s1_id", "id", "entity_id"], required=True,
        ),
        "matches": find_header_column(
            headers,
            ["matched_entity_ids", "matched_entity_id", "matched_ids", "matched_id",
             "match_ids", "match_id", "matches", "source2_ids", "source2_id",
             "source_2_ids", "source_2_id", "target_id"],
            required=True,
        ),
    }


def find_file(namelist, must_contain_all, must_contain_any_of=None):
    candidates = []
    for name in namelist:
        low = name.lower()
        if low.endswith((".tsv", ".csv")) and all(tok.lower() in low for tok in must_contain_all):
            if must_contain_any_of is None or any(tok.lower() in low for tok in must_contain_any_of):
                candidates.append(name)
    if not candidates:
        return None
    candidates.sort(key=lambda x: (len(Path(x).parts), len(x)))
    return candidates[0]


def detect_delimiter(z, filename):
    with z.open(filename) as f:
        sample = f.read(64 * 1024).decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
        return dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        return "\t" if "\t" in first_line else ","


def read_header(z, filename):
    sep = detect_delimiter(z, filename)
    with z.open(filename) as f:
        return list(pd.read_csv(f, sep=sep, nrows=0).columns)


def read_selected(z, filename, mapping):
    sep = detect_delimiter(z, filename)
    usecols = [mapping["id"], mapping["name"]]
    if mapping["address"] is not None:
        usecols.append(mapping["address"])
    if mapping["country"] is not None:
        usecols.append(mapping["country"])
    with z.open(filename) as f:
        df = pd.read_csv(
            f, sep=sep, usecols=list(dict.fromkeys(usecols)),
            dtype=str, keep_default_na=False, low_memory=False,
        )
    rename = {mapping["id"]: "_id", mapping["name"]: "_name"}
    if mapping["address"] is not None:
        rename[mapping["address"]] = "_address"
    if mapping["country"] is not None:
        rename[mapping["country"]] = "_country"
    df = df.rename(columns=rename)
    if "_address" not in df.columns:
        df["_address"] = ""
    if "_country" not in df.columns:
        df["_country"] = ""
    df["_id"] = df["_id"].astype(str).str.strip()
    return df[["_id", "_name", "_address", "_country"]].reset_index(drop=True)


def resolve_zip_path():
    if ZIP_PATH.exists():
        return ZIP_PATH
    downloads = Path.home() / "Downloads"
    zips = sorted(downloads.glob("*.zip"))
    preferred = [p for p in zips if "student_resource" in p.name.lower()]
    if len(preferred) == 1:
        print(f"Configured ZIP not found; auto-selected: {preferred[0]}")
        return preferred[0]
    if len(zips) == 1:
        print(f"Configured ZIP not found; auto-selected the only ZIP: {zips[0]}")
        return zips[0]
    names = [p.name for p in (preferred or zips)]
    raise FileNotFoundError(
        f"ZIP file not found at {ZIP_PATH}. Put the challenge ZIP in Downloads "
        f"or edit ZIP_PATH. ZIP files currently visible: {names[:20]}"
    )


def memory_status():
    if psutil is None:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def require_memory_for_array(n_rows, n_cols, dtype=np.float32, label="array"):
    available = memory_status()
    if available is None:
        return
    required = int(n_rows) * int(n_cols) * np.dtype(dtype).itemsize
    if required > available * MEMORY_SAFETY_FRACTION:
        raise MemoryError(
            f"{label} would require about {required / 2**30:.2f} GiB, while only "
            f"{available / 2**30:.2f} GiB RAM is currently available. "
            f"Lower MAX_CANDIDATES_PER_S1 / BUCKET_CAP or close other memory-heavy apps."
        )


# STAGE 1 - LOAD DATA

def load_all_data():
    banner("STAGE 1 - LOADING TRAIN DATA ONLY")

    zip_path = resolve_zip_path()
    with zipfile.ZipFile(zip_path, "r") as z:
        namelist = z.namelist()

        train_s1_f = find_file(namelist, ["train"], ["source1", "source_1"])
        train_s2_f = find_file(namelist, ["train"], ["source2", "source_2"])
        train_s3_f = find_file(namelist, ["train"], ["source3", "source_3"])
        train_gt_f = find_file(namelist, ["train"], ["ground_truth", "groundtruth", "truth"])

        test_s1_f = find_file(namelist, ["test"], ["source1", "source_1"])
        test_s2_f = find_file(namelist, ["test"], ["source2", "source_2"])
        test_s3_f = find_file(namelist, ["test"], ["source3", "source_3"])

        missing_train = [
            name for name, f in [
                ("train source1", train_s1_f), ("train source2", train_s2_f),
                ("train source3", train_s3_f), ("train ground truth", train_gt_f),
            ] if f is None
        ]
        if missing_train:
            raise FileNotFoundError(
                f"Could not locate inside the zip: {missing_train}. "
                f"Files seen: {namelist[:20]}{'...' if len(namelist) > 20 else ''}"
            )

        print("Train files:")
        print(" ", train_s1_f)
        print(" ", train_s2_f)
        print(" ", train_s3_f)
        print(" ", train_gt_f)

        s1_map = choose_source_columns(read_header(z, train_s1_f))
        s2_map = choose_source_columns(read_header(z, train_s2_f))
        s3_map = choose_source_columns(read_header(z, train_s3_f))
        gt_map = choose_gt_columns(read_header(z, train_gt_f))

        train_s1 = read_selected(z, train_s1_f, s1_map)
        train_s2 = read_selected(z, train_s2_f, s2_map)
        train_s3 = read_selected(z, train_s3_f, s3_map)
        train_s23 = pd.concat([train_s2, train_s3], ignore_index=True)
        del train_s2, train_s3

        with z.open(train_gt_f) as f:
            gt_sep = detect_delimiter(z, train_gt_f)
            gt = pd.read_csv(
                f, sep=gt_sep, usecols=[gt_map["s1"], gt_map["matches"]],
                dtype=str, keep_default_na=False, low_memory=False,
            )
        gt = gt.rename(columns={gt_map["s1"]: "_gt_s1", gt_map["matches"]: "_gt_matches"})
        gt["_gt_s1"] = gt["_gt_s1"].astype(str).str.strip()
        gt["_gt_matches"] = gt["_gt_matches"].astype(str).str.strip()

    for label, df in [("train source1", train_s1), ("train source2+3", train_s23)]:
        dup = int(df["_id"].duplicated().sum())
        if dup:
            raise ValueError(
                f"{label} has {dup:,} duplicate IDs. Entity resolution requires "
                "unique source IDs; continuing would make the submission ambiguous."
            )

    print(f"\ntrain S1  : {len(train_s1):,} rows")
    print(f"train S23 : {len(train_s23):,} rows")
    print(f"train GT  : {len(gt):,} rows")

    if test_s1_f and test_s2_f and test_s3_f:
        print("\nTest files found and will be loaded AFTER training to save RAM:")
        print(" ", test_s1_f)
        print(" ", test_s2_f)
        print(" ", test_s3_f)
    else:
        print(
            "\nWARNING: no complete test/ source set was found inside the zip. "
            "The script will not produce a real test submission."
        )
        test_s1_f = test_s2_f = test_s3_f = None

    return {
        "zip_path": zip_path,
        "test_s1_f": test_s1_f, "test_s2_f": test_s2_f, "test_s3_f": test_s3_f,
        "train_s1": train_s1, "train_s23": train_s23, "gt": gt,
        "gt_col_s1": gt_map["s1"], "gt_col_matches": gt_map["matches"],
    }


def load_test_data(data):
    if data["test_s1_f"] is None:
        return None, None

    section("LOADING TEST DATA (train data stays in memory; test loaded only now)")
    with zipfile.ZipFile(data["zip_path"], "r") as z:
        s1_f = data["test_s1_f"]
        s2_f = data["test_s2_f"]
        s3_f = data["test_s3_f"]
        test_s1 = read_selected(z, s1_f, choose_source_columns(read_header(z, s1_f)))
        test_s2 = read_selected(z, s2_f, choose_source_columns(read_header(z, s2_f)))
        test_s3 = read_selected(z, s3_f, choose_source_columns(read_header(z, s3_f)))
        test_s23 = pd.concat([test_s2, test_s3], ignore_index=True)
        del test_s2, test_s3

    for label, df in [("test source1", test_s1), ("test source2+3", test_s23)]:
        dup = int(df["_id"].duplicated().sum())
        if dup:
            raise ValueError(
                f"{label} has {dup:,} duplicate IDs. Continuing would make "
                "predicted match lists ambiguous."
            )

    print(f"test S1  : {len(test_s1):,} rows")
    print(f"test S23 : {len(test_s23):,} rows")
    return test_s1, test_s23


# STAGE 2 - NORMALIZATION

SYNONYMS = {
    "incorporated": "inc", "corporation": "corp", "corporate": "corp",
    "limited": "ltd", "company": "co", "private": "pvt",
    "international": "intl", "street": "st", "road": "rd", "av": "ave",
    "avenue": "ave", "boulevard": "blvd", "highway": "hwy",
    "apartment": "apt", "building": "bldg", "number": "no",
}
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(value):
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = _PUNCT_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    if not value:
        return ""
    return " ".join(SYNONYMS.get(tok, tok) for tok in value.split())


def digits_only(value):
    return "".join(ch for ch in str(value) if ch.isdigit())


def add_normalized_columns(df, prefix):
    section(f"Normalizing text ({prefix}, {len(df):,} rows)")
    names = df["_name"].map(normalize_text)
    addrs = df["_address"].map(normalize_text)
    countries = df["_country"].map(normalize_text)

    df[f"{prefix}_name"] = names
    df[f"{prefix}_name_compact"] = names.str.replace(" ", "", regex=False)
    df[f"{prefix}_name_first2"] = names.map(lambda x: " ".join(x.split()[:2]))
    df[f"{prefix}_name_prefix6"] = df[f"{prefix}_name_compact"].str.slice(0, 6)
    df[f"{prefix}_name_signature"] = names.map(lambda x: " ".join(sorted(set(x.split()))))
    df[f"{prefix}_address"] = addrs
    df[f"{prefix}_address_digits"] = addrs.map(digits_only)
    df[f"{prefix}_country"] = countries
    return df.drop(columns=["_name", "_address", "_country"])


# STAGE 3 - BLOCKING

BLOCK_DEFS = [
    ("name_exact", "name", False),
    ("compact_exact", "name_compact", False),
    ("name_country", "name", True),
    ("first2_country", "name_first2", True),
    ("prefix6_country", "name_prefix6", True),
    ("signature_country", "name_signature", True),
    ("addr_digits_country", "address_digits", True),
]


def build_block_key(df, prefix, col, use_country):
    base = df[f"{prefix}_{col}"]
    if use_country:
        key = base + "||" + df[f"{prefix}_country"]
        valid = (base != "") & (df[f"{prefix}_country"] != "")
    else:
        key = base
        valid = base != ""
    return key.where(valid, other=pd.NA)


def _trim_pairs_per_s1(pairs, s1, s23, limit):
    if pairs.empty:
        return pairs
    counts = pairs.groupby("s1_idx", sort=False).size()
    if not (counts > limit).any():
        return pairs.reset_index(drop=True)

    out = (
        pairs.groupby("s1_idx", sort=False, group_keys=False)
        .head(limit)
        .reset_index(drop=True)
    )
    return out


def generate_candidates(s1, s23, label="", require_nonempty=False):
    banner(f"STAGE 3 - BLOCKING {label}")

    s1 = s1.reset_index(drop=True)
    s23 = s23.reset_index(drop=True)
    current = pd.DataFrame({
        "s1_idx": np.empty(0, dtype=np.int32),
        "s23_idx": np.empty(0, dtype=np.int32),
    })

    for name, col, use_country in BLOCK_DEFS:
        key_a = build_block_key(s1, "s1", col, use_country).rename("_key")
        key_b = build_block_key(s23, "s23", col, use_country).rename("_key")

        a = pd.DataFrame({
            "_key": key_a,
            "_a": np.arange(len(s1), dtype=np.int32),
        }).dropna()
        b = pd.DataFrame({
            "_key": key_b,
            "_b": np.arange(len(s23), dtype=np.int32),
        }).dropna()

        if a.empty or b.empty:
            print(f"  {name:<20}: skipped (no usable keys)")
            del a, b, key_a, key_b
            gc.collect()
            continue

        a = a.groupby("_key", sort=False).head(BUCKET_CAP)
        b = b.groupby("_key", sort=False).head(BUCKET_CAP)

        a_counts = a.groupby("_key", sort=False).size()
        b_counts = b.groupby("_key", sort=False).size()
        common = a_counts.index.intersection(b_counts.index)
        if len(common):
            pair_estimate = a_counts.loc[common] * b_counts.loc[common]
            bad_keys = pair_estimate[pair_estimate > MAX_PAIRS_PER_KEY].index
            if len(bad_keys):
                bad_set = set(bad_keys.tolist())
                a = a[~a["_key"].isin(bad_set)]
                b = b[~b["_key"].isin(bad_set)]
                print(
                    f"  {name:<20}: dropped {len(bad_keys):,} overly generic key(s)"
                )
        del a_counts, b_counts, common

        if a.empty or b.empty:
            print(f"  {name:<20}: skipped (nothing left after filtering)")
            del a, b, key_a, key_b
            gc.collect()
            continue

        a_counts = a.groupby("_key", sort=False).size()
        b_counts = b.groupby("_key", sort=False).size()
        estimated_pairs = int(
            (a_counts * b_counts.reindex(a_counts.index, fill_value=0)).sum()
        )
        del a_counts, b_counts

        available = memory_status()
        if available is not None:
            estimated_bytes = estimated_pairs * 24
            if estimated_bytes > available * 0.50:
                print(
                    f"  {name:<20}: skipped; estimated join {estimated_pairs:,} pairs "
                    f"would be unsafe with {available / 2**30:.2f} GiB free RAM."
                )
                del a, b, key_a, key_b
                gc.collect()
                continue

        merged = a.merge(b, on="_key", how="inner", sort=False)
        block_pairs = pd.DataFrame({
            "s1_idx": merged["_a"].to_numpy(dtype=np.int32, copy=True),
            "s23_idx": merged["_b"].to_numpy(dtype=np.int32, copy=True),
        })
        print(
            f"  {name:<20}: {len(block_pairs):,} raw pairs from "
            f"{a['_key'].nunique():,} keys"
        )

        block_pairs = _trim_pairs_per_s1(
            block_pairs, s1, s23, MAX_CANDIDATES_PER_S1
        )

        if current.empty:
            current = block_pairs
        else:
            combined = pd.concat([current, block_pairs], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["s1_idx", "s23_idx"], ignore_index=True
            )
            current = _trim_pairs_per_s1(
                combined, s1, s23, MAX_CANDIDATES_PER_S1
            )
            del combined

        print(f"  {name:<20}: retained {len(block_pairs):,}; global pool {len(current):,}")

        del a, b, merged, block_pairs, key_a, key_b
        gc.collect()

    if current.empty:
        if require_nonempty:
            raise RuntimeError("No candidate pairs were generated by any blocking key.")
        print("No candidate pairs generated; returning an empty candidate set.")
        return current

    current = _trim_pairs_per_s1(current, s1, s23, MAX_CANDIDATES_PER_S1)
    print(f"\nFinal candidate pairs: {len(current):,}")
    return current


def split_match_ids(value):
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    if value[:1] in "[({":
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
    value = value.strip("[](){}").replace('"', "").replace("'", "")
    parts = re.split(r"[,;|]", value)
    if len(parts) == 1 and re.search(r"\s", value):
        parts = value.split()
    return [x.strip() for x in parts if x.strip()]


def build_truth_lookup(gt, s1_ids, s23_ids):
    truth = {}
    for s1_id, raw in zip(gt["_gt_s1"], gt["_gt_matches"]):
        matches = set(split_match_ids(raw))
        truth.setdefault(s1_id, set()).update(matches)
    for s1_id in s1_ids:
        truth.setdefault(s1_id, set())
    return truth


# STAGE 4 - FEATURES

def compute_features(pairs, s1, s23):
    n = len(pairs)
    require_memory_for_array(n, len(FEATURE_NAMES), np.float32, "feature matrix X")
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)

    s1_name = s1["s1_name"].to_numpy(dtype=object)
    s1_name_compact = s1["s1_name_compact"].to_numpy(dtype=object)
    s1_addr = s1["s1_address"].to_numpy(dtype=object)
    s1_addr_digits = s1["s1_address_digits"].to_numpy(dtype=object)
    s1_country = s1["s1_country"].to_numpy(dtype=object)
    s1_sig = s1["s1_name_signature"].to_numpy(dtype=object)

    s23_name = s23["s23_name"].to_numpy(dtype=object)
    s23_name_compact = s23["s23_name_compact"].to_numpy(dtype=object)
    s23_addr = s23["s23_address"].to_numpy(dtype=object)
    s23_addr_digits = s23["s23_address_digits"].to_numpy(dtype=object)
    s23_country = s23["s23_country"].to_numpy(dtype=object)
    s23_sig = s23["s23_name_signature"].to_numpy(dtype=object)

    s1_idx = pairs["s1_idx"].to_numpy()
    s23_idx = pairs["s23_idx"].to_numpy()

    for start in range(0, n, FEATURE_BATCH):
        stop = min(start + FEATURE_BATCH, n)
        a = s1_idx[start:stop]
        b = s23_idx[start:stop]

        na = s1_name[a]
        nb = s23_name[b]
        aa = s1_addr[a]
        ab = s23_addr[b]

        X[start:stop, 0] = [fuzz.ratio(x, y) / 100.0 for x, y in zip(na, nb)]
        X[start:stop, 1] = [fuzz.token_sort_ratio(x, y) / 100.0 for x, y in zip(na, nb)]
        X[start:stop, 2] = [fuzz.token_set_ratio(x, y) / 100.0 for x, y in zip(na, nb)]
        X[start:stop, 3] = [fuzz.ratio(x, y) / 100.0 for x, y in zip(aa, ab)]
        X[start:stop, 4] = [fuzz.token_set_ratio(x, y) / 100.0 for x, y in zip(aa, ab)]

        X[start:stop, 5] = (
            (s1_name_compact[a] != "") & (s1_name_compact[a] == s23_name_compact[b])
        ).astype(np.float32)
        X[start:stop, 6] = (
            (s1_addr_digits[a] != "") & (s1_addr_digits[a] == s23_addr_digits[b])
        ).astype(np.float32)
        X[start:stop, 7] = (
            (s1_country[a] != "") & (s1_country[a] == s23_country[b])
        ).astype(np.float32)
        X[start:stop, 8] = (
            (s1_sig[a] != "") & (s1_sig[a] == s23_sig[b])
        ).astype(np.float32)

        len_a = np.fromiter((len(x) for x in na), dtype=np.float32, count=len(na))
        len_b = np.fromiter((len(x) for x in nb), dtype=np.float32, count=len(nb))
        max_len = np.maximum(len_a, len_b)
        X[start:stop, 9] = np.where(max_len == 0, 0.0, 1.0 - np.abs(len_a - len_b) / max_len)

        if stop % (FEATURE_BATCH * 5) == 0 or stop == n:
            print(f"  features: {stop:,}/{n:,}")

    return X


# STAGE 5/6 - TRAIN MODEL + PICK THRESHOLD

def auto_tune_for_available_ram():
    global BUCKET_CAP, MAX_PAIRS_PER_KEY, MAX_CANDIDATES_PER_S1, FEATURE_BATCH
    available = memory_status()
    if available is None:
        return
    gib = available / 2**30
    if gib < 1.0:
        BUCKET_CAP = min(BUCKET_CAP, 50)
        MAX_PAIRS_PER_KEY = min(MAX_PAIRS_PER_KEY, 1500)
        MAX_CANDIDATES_PER_S1 = min(MAX_CANDIDATES_PER_S1, 2)
        FEATURE_BATCH = min(FEATURE_BATCH, 25_000)
        print("LOW-RAM MODE: <1 GiB available; using very conservative blocking.")
    elif gib < 2.0:
        BUCKET_CAP = min(BUCKET_CAP, 75)
        MAX_PAIRS_PER_KEY = min(MAX_PAIRS_PER_KEY, 3000)
        MAX_CANDIDATES_PER_S1 = min(MAX_CANDIDATES_PER_S1, 4)
        FEATURE_BATCH = min(FEATURE_BATCH, 25_000)
        print("LOW-RAM MODE: <2 GiB available; using conservative blocking.")
    elif gib < 3.0:
        BUCKET_CAP = min(BUCKET_CAP, 100)
        MAX_PAIRS_PER_KEY = min(MAX_PAIRS_PER_KEY, 5000)
        MAX_CANDIDATES_PER_S1 = min(MAX_CANDIDATES_PER_S1, 6)
        FEATURE_BATCH = min(FEATURE_BATCH, 50_000)
        print("RAM-SAFE MODE: <3 GiB available; reducing candidate pool.")
    print(f"Effective settings: BUCKET_CAP={BUCKET_CAP}, MAX_PAIRS_PER_KEY={MAX_PAIRS_PER_KEY}, "
          f"MAX_CANDIDATES_PER_S1={MAX_CANDIDATES_PER_S1}, FEATURE_BATCH={FEATURE_BATCH}")


def make_model():
    if HAVE_LIGHTGBM:
        return LGBMClassifier(
            objective="binary", n_estimators=250, learning_rate=0.05,
            num_leaves=31, max_depth=8, min_child_samples=30,
            subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=0.5, random_state=RANDOM_SEED,
            n_jobs=(1 if (memory_status() is not None and memory_status() < 3 * 2**30) else 2), verbosity=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=8,
        l2_regularization=0.5, random_state=RANDOM_SEED,
    )


def metrics_at_threshold(y, probs, threshold):
    pred = probs >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    beta = 0.5
    f05 = ((1 + beta**2) * precision * recall / (beta**2 * precision + recall)
           if (precision + recall) else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "f0_5": f05}


def train_and_tune(pairs, s1, s23, truth_lookup):
    banner("STAGE 5 - LABELING + TRAIN/VALIDATION SPLIT")

    s1_ids = s1["_id"].to_numpy()
    s23_ids = s23["_id"].to_numpy()

    s1_idx = pairs["s1_idx"].to_numpy()
    s23_idx = pairs["s23_idx"].to_numpy()

    y = np.fromiter(
        (1 if s23_ids[b] in truth_lookup.get(s1_ids[a], ()) else 0
         for a, b in zip(s1_idx, s23_idx)),
        dtype=np.uint8, count=len(pairs),
    )
    print(f"Candidate pairs      : {len(pairs):,}")
    print(f"Positive pairs found : {int(y.sum()):,}")

    total_truth_pairs = sum(len(v) for v in truth_lookup.values())
    if total_truth_pairs:
        recall_ceiling = y.sum() / total_truth_pairs
        print(f"Ground-truth pairs   : {total_truth_pairs:,}")
        print(f"Blocking recall ceiling (max possible recall given these "
              f"candidates): {recall_ceiling:.2%}")
        if recall_ceiling < 0.9:
            print(
                "  NOTE: this is below 90%. Some true matches never became "
                "candidates. Raise MAX_CANDIDATES_PER_S1 / BUCKET_CAP, or add "
                "another blocking key, to recover more of them."
            )

    if y.sum() == 0:
        raise RuntimeError(
            "Zero positive candidates - blocking did not retrieve any true "
            "matches. Loosen the blocking keys/caps before training."
        )

    unique_s1 = np.unique(s1_idx)
    min_entities_for_split = 10
    if len(unique_s1) < min_entities_for_split:
        print(
            f"\nOnly {len(unique_s1)} S1 entities have candidates - too few for a clean "
            "train/validation split. Training on everything and validating on the same "
            "data (metrics below will be optimistic; treat them as a sanity check only, "
            "not a real estimate). This should not happen on the full dataset."
        )
        is_train_row = np.ones(len(pairs), dtype=bool)
    else:
        train_entities, val_entities = train_test_split(
            unique_s1, test_size=VALIDATION_ENTITY_FRACTION, random_state=RANDOM_SEED,
        )
        train_entities = set(train_entities.tolist())
        is_train_row = np.fromiter(
            (e in train_entities for e in s1_idx), dtype=bool, count=len(pairs)
        )
        if is_train_row.sum() == 0 or (~is_train_row).sum() == 0 or \
           len(np.unique(y[is_train_row])) < 2 or len(np.unique(y[~is_train_row])) < 2:
            print(
                "\nWARNING: the entity split produced a validation or training set with "
                "only one class present. Falling back to training on everything and "
                "validating on the same data."
            )
            is_train_row = np.ones(len(pairs), dtype=bool)

    train_pairs = pairs[is_train_row].reset_index(drop=True)
    train_y = y[is_train_row]
    if (~is_train_row).any():
        val_pairs = pairs[~is_train_row].reset_index(drop=True)
        val_y = y[~is_train_row]
    else:
        val_pairs = train_pairs
        val_y = train_y

    rng = np.random.default_rng(RANDOM_SEED)
    pos_pos = np.flatnonzero(train_y == 1)
    neg_pos = np.flatnonzero(train_y == 0)
    n_neg_keep = min(len(neg_pos), len(pos_pos) * NEGATIVE_PER_POSITIVE)
    if n_neg_keep < len(neg_pos):
        neg_pos = rng.choice(neg_pos, size=n_neg_keep, replace=False)
    keep = np.sort(np.concatenate([pos_pos, neg_pos]))
    train_pairs = train_pairs.iloc[keep].reset_index(drop=True)
    train_y = train_y[keep]
    print(f"Training rows (after negative downsampling): {len(train_pairs):,} "
          f"({int(train_y.sum()):,} positive)")

    banner("STAGE 6 - TRAINING MODEL" + (" (LightGBM)" if HAVE_LIGHTGBM else
                                          " (scikit-learn HistGradientBoosting - lightgbm not installed)"))
    t0 = time.time()
    X_train = compute_features(train_pairs, s1, s23)
    model = make_model()
    model.fit(X_train, train_y)
    print(f"Training done in {time.time() - t0:.1f}s")
    del X_train
    gc.collect()

    banner("STAGE 7 - THRESHOLD TUNING ON VALIDATION SPLIT")
    X_val = compute_features(val_pairs, s1, s23)
    val_probs = model.predict_proba(X_val)[:, 1].astype(np.float32)
    del X_val
    gc.collect()

    rows = [metrics_at_threshold(val_y, val_probs, t) for t in np.round(np.arange(0.05, 0.96, 0.01), 2)]
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "threshold_results.csv", index=False)

    best = table.sort_values(["f0_5", "f1"], ascending=False).iloc[0]
    threshold = float(best["threshold"])

    print(f"\nSelected threshold : {threshold:.2f}")
    print(f"Validation precision: {best['precision']:.2%}")
    print(f"Validation recall   : {best['recall']:.2%}")
    print(f"Validation F1       : {best['f1']:.2%}")
    print(f"Validation F0.5     : {best['f0_5']:.2%}")
    print(
        "\nThese are the REAL measured numbers on held-out training entities - "
        "treat them as your honest expectation for leaderboard performance, "
        "not a promise of ~100%."
    )

    del train_pairs, train_y, val_pairs, val_y, y, val_probs
    gc.collect()
    return model, threshold, dict(best)


# STAGE 8 - PREDICT + WRITE REQUIRED OUTPUT FILES

def predict_and_write(model, threshold, s1, s23, gt_col_s1, gt_col_matches, tag):
    banner(f"STAGE 8 - PREDICTING + WRITING OUTPUTS ({tag})")

    pairs = generate_candidates(s1, s23, label=f"({tag})", require_nonempty=False)

    s1_ids = s1["_id"].to_numpy(dtype=object)
    s23_ids = s23["_id"].to_numpy(dtype=object)

    if pairs.empty:
        matching_results_df = pd.DataFrame({
            gt_col_s1: s1_ids,
            gt_col_matches: [""] * len(s1),
        })
        candidate_pairs_df = pd.DataFrame({
            gt_col_s1: s1_ids,
            "candidate_ids": [""] * len(s1),
        })
        candidate_path = OUTPUT_DIR / f"candidate_pairs_{tag}.tsv"
        results_path = OUTPUT_DIR / f"matching_results_{tag}.tsv"
        candidate_pairs_df.to_csv(candidate_path, sep="\t", index=False)
        matching_results_df.to_csv(results_path, sep="\t", index=False)
        print(f"No candidates for {tag}; wrote all entities as singletons.")
        print(f"Saved: {results_path}")
        print(f"Saved: {candidate_path}")
        return matching_results_df, candidate_pairs_df

    cand_series = (
        pd.Series(s23_ids[pairs["s23_idx"].to_numpy(dtype=np.int64)])
        .groupby(pairs["s1_idx"].to_numpy(dtype=np.int64), sort=False)
        .agg(lambda ids: ",".join(ids.astype(str)))
    )
    candidate_pairs_df = pd.DataFrame({
        gt_col_s1: s1_ids,
        "candidate_ids": [cand_series.get(i, "") for i in range(len(s1))],
    })
    candidate_path = OUTPUT_DIR / f"candidate_pairs_{tag}.tsv"
    candidate_pairs_df.to_csv(candidate_path, sep="\t", index=False)
    print(f"Saved: {candidate_path}")

    is_match = np.zeros(len(pairs), dtype=bool)
    for start_i in range(0, len(pairs), FEATURE_BATCH):
        stop_i = min(start_i + FEATURE_BATCH, len(pairs))
        batch_pairs = pairs.iloc[start_i:stop_i].reset_index(drop=True)
        X_batch = compute_features(batch_pairs, s1, s23)
        probs = model.predict_proba(X_batch)[:, 1]
        is_match[start_i:stop_i] = probs >= threshold
        del batch_pairs, X_batch, probs
        if stop_i % (FEATURE_BATCH * 5) == 0 or stop_i == len(pairs):
            print(f"  prediction: {stop_i:,}/{len(pairs):,}")
        gc.collect()

    matched_s1 = pairs["s1_idx"].to_numpy(dtype=np.int64)[is_match]
    matched_s23 = pairs["s23_idx"].to_numpy(dtype=np.int64)[is_match]

    match_series = (
        pd.Series(s23_ids[matched_s23])
        .groupby(matched_s1, sort=False)
        .agg(lambda ids: ",".join(ids.astype(str)))
    )

    matching_results_df = pd.DataFrame({
        gt_col_s1: s1_ids,
        gt_col_matches: [match_series.get(i, "") for i in range(len(s1))],
    })
    results_path = OUTPUT_DIR / f"matching_results_{tag}.tsv"
    matching_results_df.to_csv(results_path, sep="\t", index=False)

    n_matched_entities = int((matching_results_df[gt_col_matches] != "").sum())
    print(f"Saved: {results_path}")
    print(f"Source-1 entities with >=1 predicted match: {n_matched_entities:,}/{len(s1):,}")
    print(f"Source-1 entities predicted as singletons : {len(s1) - n_matched_entities:,}/{len(s1):,}")

    return matching_results_df, candidate_pairs_df


# MAIN

def run_pipeline():
    t_start = time.time()
    banner("AMAZON ML CHALLENGE - ENTITY RESOLUTION PIPELINE")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Using model backend: {'LightGBM' if HAVE_LIGHTGBM else 'scikit-learn HistGradientBoosting'}")
    if psutil is not None:
        vm = psutil.virtual_memory()
        print(f"Available RAM at start: {vm.available / 2**30:.2f} GiB / {vm.total / 2**30:.2f} GiB")
    else:
        print("psutil not installed: RAM guard disabled. Install it with: pip install psutil")

    auto_tune_for_available_ram()
    data = load_all_data()
    gt_col_s1 = data["gt_col_s1"]
    gt_col_matches = data["gt_col_matches"]
    train_s1 = add_normalized_columns(data["train_s1"], "s1")
    train_s23 = add_normalized_columns(data["train_s23"], "s23")

    truth_lookup = build_truth_lookup(
        data["gt"].rename(columns={"_gt_s1": "_gt_s1", "_gt_matches": "_gt_matches"}),
        train_s1["_id"].tolist(), train_s23["_id"].tolist(),
    )

    train_pairs = generate_candidates(train_s1, train_s23, label="(train)")
    model, threshold, val_metrics = train_and_tune(train_pairs, train_s1, train_s23, truth_lookup)
    del train_pairs
    gc.collect()

    pd.DataFrame([val_metrics]).to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "feature": FEATURE_NAMES, "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
        print("\nFeature importance:")
        print(importance.to_string(index=False))

    predict_and_write(
        model, threshold, train_s1, train_s23,
        gt_col_s1, gt_col_matches, tag="TRAIN_diagnostic",
    )

    test_s1_raw, test_s23_raw = load_test_data(data)
    if test_s1_raw is not None:
        del train_s1, train_s23, data
        gc.collect()

        test_s1 = add_normalized_columns(test_s1_raw, "s1")
        test_s23 = add_normalized_columns(test_s23_raw, "s23")
        del test_s1_raw, test_s23_raw
        gc.collect()

        matching_results_df, candidate_pairs_df = predict_and_write(
            model, threshold, test_s1, test_s23,
            gt_col_s1, gt_col_matches, tag="test",
        )
        matching_results_df.to_csv(OUTPUT_DIR / "matching_results.tsv", sep="\t", index=False)
        candidate_pairs_df.to_csv(OUTPUT_DIR / "candidate_pairs.tsv", sep="\t", index=False)
        print(f"\nFinal submission files written:")
        print(f"  {OUTPUT_DIR / 'matching_results.tsv'}")
        print(f"  {OUTPUT_DIR / 'candidate_pairs.tsv'}")
    else:
        print(
            "\nNo test set was found, so only the TRAIN diagnostic outputs were written."
        )

    banner("PIPELINE COMPLETE")
    print(f"Total runtime: {(time.time() - t_start) / 60:.1f} minutes")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_pipeline()