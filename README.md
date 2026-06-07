# TRACE: Cross-Layer Trust Resolution for Robust Federated Materials Data Mining

Public artifact for the paper *"TRACE: Cross-Layer Trust Resolution for Robust
Federated Materials Data Mining"* (under review, ICDM Applied Track).

> **Status.** This repository currently provides the **OPTIMADE cross-database
> benchmark**, the **per-experiment result tables** (3-seed medians behind every
> table/figure in the paper), and the **figure-regeneration script**. The full
> TRACE engine source (model, six aggregation rules, five attacks, the
> trust-resolution control plane, and TARA) **will be released here upon
> acceptance**.

## Contents

```
data/
  optimade_fl/            # federated partitions: separate (provenance, n=3), n5/n7/n9/n11 (scaled)
  optimade_processed/     # per-source featurized CSVs (OQMD, Materials Project, Alexandria)
results_trace/            # 3-seed result CSVs behind the paper's tables/figures
gen_figures.py            # regenerate the paper figures from results_trace/ + data/
requirements.txt
LICENSE                   # MIT
```

## Benchmark

Three real materials databases queried through the OPTIMADE API (OQMD, Materials
Project, Alexandria), featurized into 147 standardized descriptors; the target
is formation energy `e_form` (eV/atom).

| Source | #samples | #formulas | mean | std | range (full) |
|---|---|---|---|---|---|
| OQMD | 5,000 | 1,739 | -1.78 | 1.50 | [-4.1, 56.1]* |
| Materials Project | 4,999 | 3,401 | -2.06 | 0.96 | [-4.3, 4.7] |
| Alexandria | 5,000 | 4,159 | -1.42 | 1.26 | [-4.3, 4.1] |

Pairwise formula overlap is low (OQMD/MP 10.5%, OQMD/Alex 6.0%, MP/Alex 14.6%),
giving a natural non-IID federated regression task. Pooled training set: 13,800
records; shared held-out test set: 1,201 records.
*OQMD's max (56.1) reflects a few high-energy entries (only 3 records >5; 99th
percentile 0.5).

Partitions:
- `data/optimade_fl/separate/` — provenance partition (one database per worker, n=3).
- `data/optimade_fl/n5,n7,n9,n11/` — scaled IID re-splits for the scalability study.

## Results

`results_trace/*.csv` contain the 3-seed (seeds 42, 43, 44) medians reported in
the paper. Inter-seed range is typically <=0.05 R^2 (<=0.09 worst case).

| File | Paper content |
|---|---|
| `robustness_n7_f{1,2}_iid.csv` | test R^2 by rule x attack at n=7 |
| `multiscale_f1_sign_flip_iid.csv` | resolved rule + R^2 across n in {3,5,7,9,11} |
| `substitution_n7_f1_sign_flip_iid.csv` | same-context substitution, all rules |
| `adaptive_n{7,9,11}_iid.csv` | TARA vs oracle-f vs FLTrust, with estimated f-hat |
| `adv_sweep_n7_iid.csv`, `ablation_n7_iid.csv`, `intermittent_n7_iid.csv` | adaptive adversary, ablation, intermittent adversary |
| `convergence_n7_iid.csv` | per-round training dynamics |

## Regenerate the figures

```bash
pip install -r requirements.txt   # numpy, pandas, matplotlib needed here
python gen_figures.py             # reads results_trace/ + data/optimade_processed/, writes figures/*.pdf
```

This reproduces every data figure in the paper from the released result tables
and benchmark, without the engine.

## Reproducing the results from scratch

The commands that produced `results_trace/` (via the TRACE engine, released upon
acceptance) are, for reference:

```bash
python -m trace_fl.run_matrix --preset robustness   --n 7 --f 1 --reshard iid
python -m trace_fl.run_matrix --preset multiscale    --f 1 --attack sign_flip --reshard iid
python -m trace_fl.run_matrix --preset substitution  --n 7 --f 1 --attack sign_flip --reshard iid
python -m trace_fl.run_matrix --preset adaptive --n 7 --faults 0,1,2 \
    --attack sign_flip,scale,gaussian,little_is_enough,fall_of_empires --reshard iid
```

All results are 3-seed medians on a single NVIDIA RTX 4090 Laptop GPU; one n=7
run takes ~13 s.

## License

MIT (see `LICENSE`). The OPTIMADE-sourced data is redistributed for
reproducibility; the underlying databases are publicly available via OPTIMADE.
