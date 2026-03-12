import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from openai import OpenAI
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json_repair

from dotenv import load_dotenv
load_dotenv()

def load_and_validate_data(path1: Path, path2: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    # Validate required columns
    required_cols = ['reference', 'prediction']
    for df, path in [(df1, path1), (df2, path2)]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")

    # Validate same number of rows
    if len(df1) != len(df2):
        raise ValueError(
            f"Files have different number of rows: {len(df1)} vs {len(df2)}")

    # Validate references match
    if not df1['reference'].equals(df2['reference']):
        raise ValueError("References don't match between the two files")

    return df1, df2


def create_style_comparison_prompt(reference: str, prediction1: str, prediction2: str) -> str:
    prompt = f"""You are an expert writing style analyst. Your task is to compare two text and determine which one better aligns with the WRITING STYLE (not content) of the reference text.

Focus on stylistic elements such as:
- Tone (formal/informal, serious/casual, emotional/neutral, etc.)
- Syntax (sentence structure, complexity, and length patterns)
- Lexicon (word choice, vocabulary level, and specialized terminology)
- Linguistic patterns (punctuation usage, grammar structures, rhetorical devices)

Do NOT focus on:
- Factual accuracy or content correctness
- Topic similarity

Reference Text:
{reference}

Text A:
{prediction1}

Text B:
{prediction2}

Please analyze both text and determine which one better matches the writing style of the reference.

Respond with the following JSON format:
{{
    "winner": "A" or "B" or "Tie",
    "reason": "Reason for the choice if winner is not 'Tie' else 'The two texts are equally good'"
}}"""

    return prompt


def judge_pairwise_comparison(
    client: OpenAI,
    reference: str,
    prediction1: str,
    prediction2: str
) -> Dict[str, str]:
    prompt = create_style_comparison_prompt(
        reference, prediction1, prediction2)

    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            temperature=1.0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = response.choices[0].message.content.strip()
        response_json = json_repair.loads(response_text)
        # Parse response - should be just "A" or "B"
        winner = response_json['winner']
        reason = response_json['reason']

        return {
            'winner': winner,
            'reason': reason,
            'full_response': response_text
        }

    except Exception as e:
        print(f"Error during LLM judgment: {e}")
        return {
            'winner': None,
            'full_response': f"Error: {str(e)}"
        }


def evaluate_single_example(
    client: OpenAI,
    idx: int,
    reference: str,
    prediction1: str,
    prediction2: str,
    file1_name: str,
    file2_name: str
) -> Dict:
    judgment = judge_pairwise_comparison(
        client, reference, prediction1, prediction2
    )

    return {
        'index': idx,
        'reference': reference,
        f'{file1_name}_prediction': prediction1,
        f'{file2_name}_prediction': prediction2,
        'winner': judgment['winner'],
        'reason': judgment['reason'],
        'full_response': judgment['full_response']
    }


def run_pairwise_evaluation(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    file1_name: str,
    file2_name: str,
    max_workers: int = 10
) -> pd.DataFrame:
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))

    print(
        f"Running pairwise comparison between {file1_name} and {file2_name}...")
    print(f"Total examples: {len(df1)}")
    print(f"Using {max_workers} parallel workers\n")

    results = []

    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {}
        for idx in range(len(df1)):
            reference = df1.iloc[idx]['reference']
            prediction1 = df1.iloc[idx]['prediction']
            prediction2 = df2.iloc[idx]['prediction']

            future = executor.submit(
                evaluate_single_example,
                client,
                idx,
                reference,
                prediction1,
                prediction2,
                file1_name,
                file2_name
            )
            future_to_idx[future] = idx

        # Collect results as they complete
        for future in tqdm(as_completed(future_to_idx), total=len(df1), desc="Evaluating"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                idx = future_to_idx[future]
                print(f"\nError processing example {idx}: {e}")
                # Add a placeholder result for failed examples
                results.append({
                    'index': idx,
                    'reference': df1.iloc[idx]['reference'],
                    f'{file1_name}_prediction': df1.iloc[idx]['prediction'],
                    f'{file2_name}_prediction': df2.iloc[idx]['prediction'],
                    'winner': None,
                    'full_response': f"Error: {str(e)}"
                })

    # Sort results by index to maintain original order
    results.sort(key=lambda x: x['index'])

    return pd.DataFrame(results)


def print_summary_statistics(results_df: pd.DataFrame, file1_name: str, file2_name: str):
    print("\n" + "=" * 60)
    print("PAIRWISE COMPARISON SUMMARY")
    print("=" * 60)

    total = len(results_df)
    a_wins = (results_df['winner'] == 'A').sum()
    b_wins = (results_df['winner'] == 'B').sum()
    ties_errors = total - a_wins - b_wins

    print(f"\nTotal comparisons: {total}")
    print(f"\n{file1_name} (A) wins: {a_wins} ({a_wins/total*100:.1f}%)")
    print(f"{file2_name} (B) wins: {b_wins} ({b_wins/total*100:.1f}%)")
    print(f"Ties/Errors: {ties_errors} ({ties_errors/total*100:.1f}%)")

    if a_wins > b_wins:
        print(f"\n{file1_name} has better style alignment overall")
    elif b_wins > a_wins:
        print(f"\n{file2_name} has better style alignment overall")
    else:
        print(f"\nBoth perform similarly")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Use LLM to judge which predictions better align with reference writing style"
    )
    parser.add_argument(
        '--path1',
        type=str,
        required=True,
        help='Path to first CSV file with predictions'
    )
    parser.add_argument(
        '--path2',
        type=str,
        required=True,
        help='Path to second CSV file with predictions'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='llm_judge_results.csv',
        help='Output path for results CSV (default: llm_judge_results.csv)'
    )

    args = parser.parse_args()

    path1 = Path(args.path1)
    path2 = Path(args.path2)

    # Extract filenames (without extension) for labeling
    file1_name = path1.stem
    file2_name = path2.stem

    # Validate files exist
    if not path1.exists():
        raise FileNotFoundError(f"File not found: {path1}")
    if not path2.exists():
        raise FileNotFoundError(f"File not found: {path2}")

    # Load and validate data
    print("Loading data...")
    df1, df2 = load_and_validate_data(path1, path2)
    print(f"✓ Loaded {len(df1)} examples from both files")
    print(f"✓ References match between files\n")

    # Run pairwise evaluation
    results_df = run_pairwise_evaluation(
        df1, df2,
        file1_name,
        file2_name
    )

    # Save results
    output_path = Path(args.output)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")

    # Print summary
    print_summary_statistics(results_df, file1_name, file2_name)


if __name__ == '__main__':
    main()
