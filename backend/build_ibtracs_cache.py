import json
import os

from backend import ibtracs_common as C

ROOT = C.project_root()
df = C.clean_ibtracs(os.path.join(ROOT, "data", "ibtracs.csv"))
df = C.add_sequence_features(df)
cache = C.build_storm_cache(df, ["AMPHAN", "BIPAR", "FANI", "TAUKT", "YAAS", "TITLI", "PHAILIN"])
with open(os.path.join(ROOT, "models", "ibtracs_ni_storms.json"), "w") as f:
    json.dump(cache, f, indent=2)
for k, v in cache.items():
    w = v["last_row"]["WIND"]
    print(f"{k:28s} peak {w:6.1f} kt  pts {len(v['track']):3d}")