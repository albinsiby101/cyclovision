"""
IMD cyclone category wind bands (knots), shared by the API and the training
pipelines.

This module deliberately has NO third-party imports: the FastAPI runtime pulls
CAT_BANDS from here, and importing it from backend/ibtracs_common.py would drag
pandas (a ~36 MB resident import) into the cloud process for the sake of a
literal. Single source of truth for both the training scripts and the API.
"""

CAT_BANDS = [("Depression", 17), ("Deep Depression", 28), ("Cyclonic Storm", 34),
             ("Severe Cyclonic Storm", 48), ("Very Severe Cyclonic Storm", 64),
             ("Extremely Severe Cyclonic Storm", 90), ("Super Cyclonic Storm", 120)]


def imd_category(wind) -> str:
    label = CAT_BANDS[0][0]
    for name, lo in CAT_BANDS:
        if wind >= lo:
            label = name
    return label
