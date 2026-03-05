# python eval_scenario.py --batch \
#       --results_dir outputs/genre_holdout_exp \
#       --output_csv outputs/genre_holdout_exp/eval_results/eval_results_rating.csv \
#       --model gpt-5-mini

python eval_pairwise.py --batch \
        --results_dir outputs/genre_holdout_exp \
        --output_csv outputs/genre_holdout_exp/eval_results/eval_results_pairwise.csv \
        --model gpt-5-mini