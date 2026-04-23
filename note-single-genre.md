After executing "python eval_scenario.py --batch \
        --results_dir outputs/single_genre_prompt_exp/gpt-5.2 \
        --output_csv outputs/single_genre_prompt_exp/gpt-5.2/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2", I got:
```
============================================================
SUMMARY: Mean style-match rating by condition (1-7 scale)
============================================================
  seen    : mean=3.04 +/- 0.06  (n=180)
  unseen  : mean=3.11 +/- 0.02  (n=3384)

Per-author breakdown:
  Author    Seen mean  Unseen mean    Delta
  ------------------------------------------
  a0             2.50         3.10    -0.60
  a1             3.00         3.11    -0.11
  a2             3.11         3.49    -0.38
  a3             3.00         2.94    +0.06
  a4             2.94         2.87    +0.08
  a5             2.39         2.93    -0.54
  a6             3.06         2.93    +0.13
  a7             3.33         3.27    +0.07
  a8             3.78         3.24    +0.54
  a9             3.33         3.43    -0.09

  Avg delta (seen - unseen): -0.09 +/- 0.11
  Paired t-test: t=-0.785, p=0.4524
  => Not significant (p >= 0.05)
```

-------


After executing "python eval_pairwise.py --batch \
        --results_dir outputs/single_genre_prompt_exp/gpt-5.2 \
        --output_csv outputs/single_genre_prompt_exp/gpt-5.2/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2", I got:
```
============================================================
BATCH SUMMARY: Pairwise win rates (paper-style)
============================================================

Overall (n=678):
  Seen wins:   145/678 (21.4%)
  Unseen wins: 138/678 (20.4%)
  Ties:        395/678 (58.3%)

Per unseen variant:
  Variant               Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------------------
  genre-b                   29.5%        14.7%    55.8%   129
  genre-c                   34.1%        19.6%    46.4%   138
  genre-d                   14.4%        23.5%    62.1%   132
  genre-e                    6.1%        28.6%    65.3%   147
  genre-s                   24.2%        14.4%    61.4%   132

Per author:
  Author    Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------
  a0             1.4%        47.2%    51.4%    72
  a1            23.6%        30.6%    45.8%    72
  a2            10.0%        33.3%    56.7%    60
  a3            16.7%         9.7%    73.6%    72
  a4            33.3%        16.7%    50.0%    72
  a5             6.9%        27.8%    65.3%    72
  a6            43.1%        16.7%    40.3%    72
  a7            22.2%         0.0%    77.8%    54
  a8            31.9%        13.9%    54.2%    72
  a9            23.3%         1.7%    75.0%    60

  Mean seen win rate: 21.2% +/- 4.1%
  One-sample t-test (H0: win rate = 50%): t=-7.058, p=0.0001
  => Significant difference (p < 0.05)
```

-------



After executing "python eval_scenario.py --batch \
        --results_dir outputs/single_genre_rag_exp/gpt-5.2 \
        --output_csv outputs/single_genre_rag_exp/gpt-5.2/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2", I got:
```
============================================================
SUMMARY: Mean style-match rating by condition (1-7 scale)
============================================================
  seen    : mean=5.13 +/- 0.07  (n=180)
  unseen  : mean=4.17 +/- 0.02  (n=3600)

Per-author breakdown:
  Author    Seen mean  Unseen mean    Delta
  ------------------------------------------
  a0             4.89         4.44    +0.45
  a1             5.44         4.00    +1.44
  a2             5.28         4.46    +0.82
  a3             4.83         3.91    +0.92
  a4             4.94         3.98    +0.96
  a5             4.67         3.93    +0.74
  a6             5.22         3.91    +1.32
  a7             5.33         4.44    +0.89
  a8             5.33         4.32    +1.01
  a9             5.39         4.36    +1.03

  Avg delta (seen - unseen): +0.96 +/- 0.09
  Paired t-test: t=10.850, p=0.0000
  => Significant difference (p < 0.05)
```

-------




After executing "python eval_pairwise.py --batch \
        --results_dir outputs/single_genre_rag_exp/gpt-5.2 \
        --output_csv outputs/single_genre_rag_exp/gpt-5.2/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2", I got:
```
============================================================
BATCH SUMMARY: Pairwise win rates (paper-style)
============================================================

Overall (n=720):
  Seen wins:   415/720 (57.6%)
  Unseen wins: 33/720 (4.6%)
  Ties:        272/720 (37.8%)

Per unseen variant:
  Variant               Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------------------
  genre-b                   64.5%         1.4%    34.0%   141
  genre-c                   55.1%         7.2%    37.7%   138
  genre-d                   56.9%         4.2%    38.9%   144
  genre-e                   44.9%         8.2%    46.9%   147
  genre-s                   66.7%         2.0%    31.3%   150

Per author:
  Author    Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------
  a0            54.2%        16.7%    29.2%    72
  a1            75.0%         2.8%    22.2%    72
  a2            40.3%         5.6%    54.2%    72
  a3            59.7%         6.9%    33.3%    72
  a4            54.2%         1.4%    44.4%    72
  a5            51.4%         4.2%    44.4%    72
  a6            72.2%         2.8%    25.0%    72
  a7            54.2%         0.0%    45.8%    72
  a8            40.3%         4.2%    55.6%    72
  a9            75.0%         1.4%    23.6%    72

  Mean seen win rate: 57.6% +/- 4.1%
  One-sample t-test (H0: win rate = 50%): t=1.872, p=0.0939
  => Not significant (p >= 0.05)
```

-------


After executing "python eval_scenario.py --batch \
        --results_dir outputs/single_genre_steering_exp/mistral-7b \
        --output_csv outputs/single_genre_steering_exp/mistral-7b/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2", I got:

```
============================================================
SUMMARY: Mean style-match rating by condition (1-7 scale)
============================================================
  seen    : mean=2.12 +/- 0.03  (n=660)
  unseen  : mean=1.97 +/- 0.01  (n=4500)

Per-author breakdown:
  Author    Seen mean  Unseen mean    Delta
  ------------------------------------------
  a0             2.26         2.10    +0.16
  a1             2.09         1.94    +0.15
  a2             2.30         2.00    +0.30
  a3             1.92         1.74    +0.18
  a4             2.47         2.37    +0.10
  a5             1.76         1.89    -0.13
  a6             2.27         2.10    +0.17
  a7             2.15         1.99    +0.16
  a8             1.85         1.65    +0.20
  a9             2.08         1.86    +0.21

  Avg delta (seen - unseen): +0.15 +/- 0.04
  Paired t-test: t=4.216, p=0.0023
  => Significant difference (p < 0.05)
```

-------


After executing "python eval_pairwise.py --batch \
        --results_dir outputs/single_genre_steering_exp/mistral-7b \
        --output_csv outputs/single_genre_steering_exp/mistral-7b/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2", I got:

```
============================================================
BATCH SUMMARY: Pairwise win rates (paper-style)
============================================================

Overall (n=3300):
  Seen wins:   814/3300 (24.7%)
  Unseen wins: 466/3300 (14.1%)
  Ties:        2020/3300 (61.2%)

Per unseen variant:
  Variant               Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------------------
  genre-b                   31.2%        11.5%    57.3%   660
  genre-c                   20.8%        13.2%    66.1%   660
  genre-d                   14.1%        22.6%    63.3%   660
  genre-e                   24.5%        15.8%    59.7%   660
  genre-s                   32.7%         7.6%    59.7%   660

Per author:
  Author    Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------
  a0            20.3%        13.9%    65.8%   330
  a1            18.8%        15.2%    66.1%   330
  a2            34.5%        13.0%    52.4%   330
  a3            37.0%        10.9%    52.1%   330
  a4            17.3%        14.8%    67.9%   330
  a5            17.9%        21.8%    60.3%   330
  a6            27.3%         8.5%    64.2%   330
  a7            22.7%        13.3%    63.9%   330
  a8            25.5%        14.2%    60.3%   330
  a9            25.5%        15.5%    59.1%   330

  Mean seen win rate: 24.7% +/- 2.1%
  One-sample t-test (H0: win rate = 50%): t=-11.796, p=0.0000
  => Significant difference (p < 0.05)
```

-------


After executing "python eval_scenario.py --batch \
        --results_dir outputs/single_genre_exp \
        --output_csv outputs/single_genre_exp/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2", I got:

```
============================================================
SUMMARY: Mean style-match rating by condition (1-7 scale)
============================================================
  seen    : mean=3.74 +/- 0.10  (n=180)
  unseen  : mean=2.89 +/- 0.03  (n=1584)

Per-author breakdown:
  Author    Seen mean  Unseen mean    Delta
  ------------------------------------------
  a0             3.44         3.44    +0.00
  a1             4.39         2.78    +1.60
  a2             3.78         2.94    +0.84
  a3             3.61         3.15    +0.46
  a4             4.39         3.22    +1.17
  a5             3.22         2.63    +0.59
  a6             3.50          nan     +nan
  a7             3.83         2.60    +1.24
  a8             3.72         2.77    +0.95
  a9             3.56         2.99    +0.57

  Avg delta (seen - unseen): +0.83 +/- 0.16
  Paired t-test: t=5.151, p=0.0009
  => Significant difference (p < 0.05)
```

-------




After executing "python eval_pairwise.py --batch \
        --results_dir outputs/single_genre_exp \
        --output_csv outputs/single_genre_exp/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2", I got:

```
============================================================
BATCH SUMMARY: Pairwise win rates (paper-style)
============================================================

Overall (n=300):
  Seen wins:   211/300 (70.3%)
  Unseen wins: 24/300 (8.0%)
  Ties:        65/300 (21.7%)

Per unseen variant:
  Variant               Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------------------
  genre-b                   61.8%        11.8%    26.5%   102
  genre-c                   82.5%         4.8%    12.7%    63
  genre-d                   77.1%         2.1%    20.8%    96
  genre-e                   33.3%        16.7%    50.0%    12
  genre-s                   66.7%        18.5%    14.8%    27

Per author:
  Author    Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------
  a0            50.0%         4.2%    45.8%    24
  a1            75.9%         9.3%    14.8%    54
  a2            70.8%        12.5%    16.7%    24
  a3            43.3%        13.3%    43.3%    30
  a4            76.7%         0.0%    23.3%    30
  a5            71.9%        12.3%    15.8%    57
  a7            88.1%         4.8%     7.1%    42
  a8            63.0%         7.4%    29.6%    27
  a9            83.3%         0.0%    16.7%    12

  Mean seen win rate: 69.2% +/- 4.9%
  One-sample t-test (H0: win rate = 50%): t=3.907, p=0.0045
  => Significant difference (p < 0.05)
```

-------





After executing "python eval_scenario.py --batch \
        --results_dir outputs/single_genre_sft_exp \
        --output_csv outputs/single_genre_sft_exp/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2", I got:

```
============================================================
SUMMARY: Mean style-match rating by condition (1-7 scale)
============================================================
  seen    : mean=4.00 +/- 0.10  (n=180)
  unseen  : mean=2.88 +/- 0.03  (n=1296)

Per-author breakdown:
  Author    Seen mean  Unseen mean    Delta
  ------------------------------------------
  a0             4.22         3.22    +1.01
  a1             4.94         2.67    +2.27
  a2             4.17          nan     +nan
  a3             4.00         2.76    +1.24
  a4             4.11         3.38    +0.74
  a5             2.89         2.82    +0.07
  a6             3.39         2.88    +0.50
  a7             3.67         2.79    +0.88
  a8             4.72         2.45    +2.27
  a9             3.89         2.93    +0.96

  Avg delta (seen - unseen): +1.10 +/- 0.25
  Paired t-test: t=4.468, p=0.0021
  => Significant difference (p < 0.05)
```

-------




After executing "python eval_pairwise.py --batch \
        --results_dir outputs/single_genre_sft_exp \
        --output_csv outputs/single_genre_sft_exp/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2", I got:

```
============================================================
BATCH SUMMARY: Pairwise win rates (paper-style)
============================================================

Overall (n=264):
  Seen wins:   189/264 (71.6%)
  Unseen wins: 13/264 (4.9%)
  Ties:        62/264 (23.5%)

Per unseen variant:
  Variant               Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------------------
  genre-c                   66.7%        16.7%    16.7%    12
  genre-d                   88.2%         2.0%     9.8%   102
  genre-e                   62.5%         6.2%    31.2%    48
  genre-s                   59.8%         5.9%    34.3%   102

Per author:
  Author    Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------
  a0            63.0%         7.4%    29.6%    27
  a1            88.9%         3.7%     7.4%    27
  a3            75.0%         0.0%    25.0%    24
  a4            63.9%        11.1%    25.0%    36
  a5            33.3%        11.1%    55.6%    18
  a6            72.9%         0.0%    27.1%    48
  a7            81.5%         0.0%    18.5%    27
  a8            90.0%         3.3%     6.7%    30
  a9            63.0%        11.1%    25.9%    27

  Mean seen win rate: 70.2% +/- 5.8%
  One-sample t-test (H0: win rate = 50%): t=3.490, p=0.0082
  => Significant difference (p < 0.05)
```

-------


After executing "python eval_scenario.py --batch \
        --results_dir outputs/genre_holdout_steering_exp/mistral-7b \
        --output_csv outputs/genre_holdout_steering_exp/mistral-7b/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2", I got:

```
============================================================
SUMMARY: Mean style-match rating by condition (1-7 scale)
============================================================
  seen    : mean=2.13 +/- 0.03  (n=660)
  unseen  : mean=1.06 +/- 0.01  (n=900)

Per-author breakdown:
  Author    Seen mean  Unseen mean    Delta
  ------------------------------------------
  a0             2.29         1.46    +0.83
  a1             2.14         1.00    +1.14
  a2             2.24         1.00    +1.24
  a3             1.85         1.03    +0.82
  a4             2.42         1.09    +1.34
  a5             1.80         1.01    +0.79
  a6             2.33         1.01    +1.32
  a7             2.24         1.02    +1.22
  a8             1.86         1.00    +0.86
  a9             2.12         1.02    +1.10

  Avg delta (seen - unseen): +1.07 +/- 0.07
  Paired t-test: t=15.371, p=0.0000
  => Significant difference (p < 0.05)
```

-------


After executing "python eval_pairwise.py --batch \
        --results_dir outputs/genre_holdout_steering_exp/mistral-7b \
        --output_csv outputs/genre_holdout_steering_exp/mistral-7b/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2", I got:

```
============================================================
BATCH SUMMARY: Pairwise win rates (paper-style)
============================================================

Overall (n=660):
  Seen wins:   425/660 (64.4%)
  Unseen wins: 30/660 (4.5%)
  Ties:        205/660 (31.1%)

Per unseen variant:
  Variant               Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------------------
  genre-b                   85.4%         1.6%    13.0%   123
  genre-c                   45.1%         9.7%    45.1%   144
  genre-d                   20.5%         9.8%    69.7%   132
  genre-e                   78.3%         0.8%    20.9%   129
  genre-s                   96.2%         0.0%     3.8%   132

Per author:
  Author    Seen win%  Unseen win%     Tie%     n
  ----------------------------------------------
  a0            40.9%         7.6%    51.5%    66
  a1            72.7%         3.0%    24.2%    66
  a2            68.2%         3.0%    28.8%    66
  a3            74.2%         3.0%    22.7%    66
  a4            53.0%         7.6%    39.4%    66
  a5            54.5%        10.6%    34.8%    66
  a6            89.4%         0.0%    10.6%    66
  a7            57.6%         1.5%    40.9%    66
  a8            68.2%         6.1%    25.8%    66
  a9            65.2%         3.0%    31.8%    66

  Mean seen win rate: 64.4% +/- 4.3%
  One-sample t-test (H0: win rate = 50%): t=3.367, p=0.0083
  => Significant difference (p < 0.05)
```

-------