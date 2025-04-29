import heapq

import numpy as np
import polars as pl
from numba import njit
from numba.types import Array, uint32, int32, float32

@njit(
    uint32[:](
        Array(uint32, 1, 'C', readonly=True), 
        Array(int32, 1, 'C', readonly=True), 
        Array(int32, 1, 'C', readonly=True), 
        Array(float32, 1, 'C', readonly=True)
    ), cache=True
)
def _collapse_hits(chrom_ids, starts, ends, similarities):
    n = chrom_ids.shape[0]
    out = np.ones(n, dtype=np.uint32)
    heap = [(np.uint32(0), np.int32(0), -1) for _ in range(0)]

    for i in range(n):
        chrom_new = chrom_ids[i]
        start_new = starts[i]
        end_new = ends[i]
        sim_new = similarities[i]

        while heap and heap[0] < (chrom_new, start_new, -1):
            heapq.heappop(heap)

        for _, _, idx in heap:
            cmp = sim_new > similarities[idx]
            out[idx] &= cmp
            out[i] &= not cmp

        heapq.heappush(heap, (chrom_new, end_new, i))

    return out


def collapse_hits(hits_df, overlap_frac):
    chroms = hits_df["chr"].unique(maintain_order=True)

    if not chroms.is_empty():
        chrom_to_id = {
            chrom: i for i, chrom in enumerate(chroms)
        }
        df = hits_df.select(
            chrom_id=pl.col("chr").replace_strict(chrom_to_id, return_dtype=pl.UInt32),
            start_trim=pl.col("start") * 2 + ((pl.col("end") - pl.col("start")) * overlap_frac).cast(pl.Int32),
            end_trim=pl.col("end") * 2 - ((pl.col("end") - pl.col("start")) * overlap_frac).cast(pl.Int32),
            similarity=pl.col("hit_similarity")
        )
    else:
        df = hits_df.select(
            chrom_id=pl.col("peak_id"),
            start_trim=pl.col("start") * 2 + ((pl.col("end") - pl.col("start")) * overlap_frac).cast(pl.Int32),
            end_trim=pl.col("end") * 2 - ((pl.col("end") - pl.col("start")) * overlap_frac).cast(pl.Int32),
            similarity=pl.col("hit_similarity")
        )

    df = df.rechunk()
    chrom_ids = df["chrom_id"].to_numpy(allow_copy=False)
    starts = df["start_trim"].to_numpy(allow_copy=False)
    ends = df["end_trim"].to_numpy(allow_copy=False)
    similarities = df["similarity"].to_numpy(allow_copy=False)
    is_primary = _collapse_hits(chrom_ids, starts, ends, similarities)

    df_out = hits_df.with_columns(
        is_primary=pl.Series(is_primary, dtype=pl.UInt32)
    )

    return df_out


def intersect_hits(hits_dfs, relaxed):
    if relaxed:
        join_cols = [
            "chr", "start_untrimmed", "end_untrimmed",
            "motif_name", "strand"
        ]
    else:
        join_cols = [
            "chr", "start", "end", "start_untrimmed", "end_untrimmed",
            "motif_name", "strand", "peak_name", "peak_id"
        ]

    if len(hits_dfs) < 1:
        raise ValueError("At least one hits dataframe required")

    hits_df = hits_dfs[0]
    for i in range(1, len(hits_dfs)):
        hits_df = hits_df.join(
            hits_dfs[i],
            on=join_cols,
            how="inner",
            suffix=f"_{i}",
            join_nulls=True,
            coalesce=True
        )

    return hits_df