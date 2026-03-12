# python eval_scenario.py --batch \
#       --results_dir outputs/genre_holdout_exp \
#       --output_csv outputs/genre_holdout_exp/eval_results/eval_results_rating.csv \
#       --model gpt-5-mini


python eval_scenario.py --batch \
      --results_dir outputs/genre_holdout_rag_exp/gpt-5.2 \
      --output_csv outputs/genre_holdout_rag_exp/gpt-5.2/eval_results/eval_results_rating.csv \
      --model gpt-5.2


python eval_scenario.py --batch \
      --results_dir outputs/genre_holdout_prompt_exp/gpt-5.2 \
      --output_csv outputs/genre_holdout_prompt_exp/gpt-5.2/eval_results/eval_results_rating.csv \
      --model gpt-5.2

# python eval_pairwise.py --batch \
#         --results_dir outputs/genre_holdout_exp \
#         --output_csv outputs/genre_holdout_exp/eval_results/eval_results_pairwise.csv \
#         --model gpt-5-mini


python eval_pairwise.py --batch \
        --results_dir outputs/genre_holdout_rag_exp/gpt-5.2 \
        --output_csv outputs/genre_holdout_rag_exp/gpt-5.2/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2

python eval_pairwise.py --batch \
        --results_dir outputs/genre_holdout_prompt_exp/gpt-5.2 \
        --output_csv outputs/genre_holdout_prompt_exp/gpt-5.2/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2


python eval_scenario.py --batch \
      --results_dir outputs/genre_holdout_steering_exp/mistral-7b \
      --output_csv outputs/genre_holdout_steering_exp/mistral-7b/eval_results/eval_results_rating.csv \
      --model gpt-5.2

python eval_pairwise.py --batch \
        --results_dir outputs/genre_holdout_steering_exp/mistral-7b \
        --output_csv outputs/genre_holdout_steering_exp/mistral-7b/eval_results/eval_results_pairwise.csv \
        --model gpt-5.2