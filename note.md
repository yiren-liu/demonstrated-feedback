After executing "python eval_scenario.py --batch       --results_dir outputs/genre_holdout_rag_exp/gpt-5.2       --output_csv outputs/genre_holdout_rag_exp/gpt-5.2/eval_results/eval_results_rating.csv       --model gpt-5.2", I got:
```
Wrote 1062 evaluation rows to outputs/genre_holdout_rag_exp/gpt-5.2/eval_results/eval_results_rating.csv

============================================================
SUMMARY: Mean style-match rating by condition (1-7 scale)
============================================================
  seen    : mean=5.21 +/- 0.07  (n=180)
  unseen  : mean=4.81 +/- 0.04  (n=882)

Per-author breakdown:
  Author    Seen mean  Unseen mean    Delta
  ------------------------------------------
  a0             4.83         5.19    -0.36
  a1             5.50         4.34    +1.16
  a2             5.56         4.91    +0.64
  a3             4.94         5.04    -0.10
  a4             5.00         4.51    +0.49
  a5             4.67         4.53    +0.13
  a6             5.28         4.57    +0.71
  a7             5.39         5.20    +0.19
  a8             5.39         4.75    +0.64
  a9             5.50         5.07    +0.43

  Avg delta (seen - unseen): +0.39 +/- 0.14
  Paired t-test: t=2.841, p=0.0194
  => Significant difference (p < 0.05)
```

-------

After executing "python eval_pairwise.py --batch         --results_dir outputs/genre_holdout_rag_exp/gpt-5.2         --output_csv outputs/genre_holdout_rag_exp/gpt-5.2/eval_results/eval_results_pairwise.csv         --model gpt-5.2", I got:
```

============================================================
BATCH SUMMARY: Pairwise win rates (paper-style)
============================================================

Overall (n=180):
  Seen wins:   65/180 (36.1%)
  Unseen wins: 13/180 (7.2%)
  Ties:        102/180 (56.7%)

Per unseen variant:
  Variant               Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------------------
  genre-b                   46.2%        15.4%    38.5%    39
  genre-c                   42.9%         4.8%    52.4%    42
  genre-d                   22.2%         0.0%    77.8%    36
  genre-e                   57.6%         0.0%    42.4%    33
  genre-s                    6.7%        16.7%    76.7%    30

Per author:
  Author    Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------
  a0            22.2%        38.9%    38.9%    18                                                                                                         Mar-26
  a1            50.0%         0.0%    50.0%    18
  a2            22.2%         5.6%    72.2%    18
  a3            27.8%         5.6%    66.7%    18
  a4            33.3%         5.6%    61.1%    18
  a5            27.8%         0.0%    72.2%    18
  a6            61.1%         5.6%    33.3%    18
  a7            27.8%         0.0%    72.2%    18
  a8            33.3%        11.1%    55.6%    18
  a9            55.6%         0.0%    44.4%    18

  Mean seen win rate: 36.1% +/- 4.5%
  One-sample t-test (H0: win rate = 50%): t=-3.101, p=0.0127
  => Significant difference (p < 0.05)
```


----


After executing "python eval_scenario.py --batch       --results_dir outputs/genre_holdout_prompt_exp/gpt-5.2       --output_csv outputs/genre_holdout_prompt_exp/gpt-5.2/eval_results/eval_results_rating.csv       --model gpt-5.2", I got:
```
Wrote 1080 evaluation rows to outputs/genre_holdout_prompt_exp/gpt-5.2/eval_results/eval_results_rating.csv

============================================================
SUMMARY: Mean style-match rating by condition (1-7 scale)
============================================================
  seen    : mean=3.04 +/- 0.06  (n=180)
  unseen  : mean=3.19 +/- 0.03  (n=900)

Per-author breakdown:
  Author    Seen mean  Unseen mean    Delta
  ------------------------------------------
  a0             2.56         3.10    -0.54
  a1             3.00         3.38    -0.38
  a2             3.17         3.51    -0.34
  a3             3.00         3.12    -0.12
  a4             3.00         2.67    +0.33
  a5             2.39         3.03    -0.64
  a6             3.11         2.90    +0.21
  a7             3.28         3.52    -0.24
  a8             3.72         3.14    +0.58
  a9             3.22         3.49    -0.27

  Avg delta (seen - unseen): -0.14 +/- 0.12
  Paired t-test: t=-1.138, p=0.2846
  => Not significant (p >= 0.05)
```


----


After executing "python eval_scenario.py --batch       --results_dir outputs/genre_holdout_prompt_exp/gpt-5.2       --output_csv outputs/genre_holdout_prompt_exp/gpt-5.2/eval_results/eval_results_rating.csv       --model gpt-5.2", I got:
```
============================================================
BATCH SUMMARY: Pairwise win rates (paper-style)
============================================================

Overall (n=180):
  Seen wins:   31/180 (17.2%)
  Unseen wins: 32/180 (17.8%)
  Ties:        117/180 (65.0%)

Per unseen variant:
  Variant               Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------------------
  genre-b                   17.9%        12.8%    69.2%    39
  genre-c                   33.3%        21.4%    45.2%    42
  genre-d                   11.1%         0.0%    88.9%    36
  genre-e                   15.2%        33.3%    51.5%    33
  genre-s                    3.3%        23.3%    73.3%    30

Per author:
  Author    Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------
  a0             5.6%        38.9%    55.6%    18
  a1             0.0%        38.9%    61.1%    18
  a2             5.6%        22.2%    72.2%    18
  a3             0.0%        22.2%    77.8%    18
  a4            66.7%         0.0%    33.3%    18
  a5            11.1%        38.9%    50.0%    18
  a6            50.0%        11.1%    38.9%    18
  a7             0.0%         0.0%   100.0%    18
  a8            16.7%         0.0%    83.3%    18
  a9            16.7%         5.6%    77.8%    18

  Mean seen win rate: 17.2% +/- 7.2%
  One-sample t-test (H0: win rate = 50%): t=-4.527, p=0.0014
  => Significant difference (p < 0.05)
```

-----