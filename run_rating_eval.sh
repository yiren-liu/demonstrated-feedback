# python eval_scenario.py --batch \
#       --results_dir outputs/scenario_experiment \
#       --output_csv outputs/scenario_experiment/eval_results_rating.csv \
#       --model gpt-5-mini

python eval_pairwise.py --batch \
        --results_dir outputs/scenario_experiment \
        --output_csv outputs/scenario_experiment/pairwise_results_rating.csv \
        --model gpt-5-mini