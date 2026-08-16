"""Shared text-level column definitions (macro / sector / related / target)."""

LEVEL_PREFIXES = [
    ("macro", ["macro_category"]),
    ("sector", ["sector_category"]),
    ("related", ["relatedCompany_category"]),
    ("target", ["targetCompany_category", "filing_", "lseg_news"]),
]

PRICE_FACTOR_NAMES = ["log_ret_close", "hl_range", "oc_range", "log_ret_volume", "mom_5d"]


def get_level_id_cols(df_cols):
    result = []
    for level_name, prefixes in LEVEL_PREFIXES:
        matched = []
        for pfx in prefixes:
            matched += [c for c in df_cols if c.startswith(pfx)]
        result.append((level_name, matched))
    return result
