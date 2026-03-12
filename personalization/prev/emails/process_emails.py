from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Optional

from pydantic import BaseModel
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - only triggered without dependency.
    raise SystemExit(
        "The `openai` package is required. Install it via `pip install openai`."
    ) from exc


# Default columns expected in the per-employee CSV files.
REQUIRED_COLUMNS = {"file", "date", "from", "to", "subject", "email"}


@dataclass
class EmailRecord:
    """Typed representation of an email row from the CSV."""

    file: str
    date: str
    sender: str
    recipient: str
    subject: str
    body: str


class PromptExtraction(BaseModel):
    """Structured response expected from GPT-4o mini."""

    key_points: list[str]
    prompt: str


class DraftEmail(BaseModel):
    """Structured response for the email generation call."""

    email: str


class _ClientPool:
    """Provide each worker thread with its own OpenAI client."""

    def __init__(self, dry_run: bool) -> None:
        self._dry_run = dry_run
        self._local: threading.local = threading.local()

    def get(self) -> Optional[OpenAI]:
        if self._dry_run:
            return None
        client = getattr(self._local, "client", None)
        if client is None:
            client = OpenAI()
            self._local.client = client
        return client


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GPT-4o mini prompts for an employee email CSV."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV file under emails_by_employee to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Directory (or explicit file path) for the augmented CSV. "
        "Defaults to the input file's directory.",
        default="dataset/emails_by_employee.prompts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of emails to process.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay between API calls to avoid rate limits (seconds).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls and emit placeholder prompts (useful for testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker threads to use. Set to 1 to disable parallelism.",
    )
    return parser.parse_args(argv)


def assert_columns(header: Iterable[str]) -> None:
    missing = REQUIRED_COLUMNS.difference({h.strip().lower() for h in header})
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")


def row_to_record(row: Dict[str, str]) -> EmailRecord:
    """Convert a CSV row dict into a typed record."""
    return EmailRecord(
        file=row.get("file", ""),
        date=row.get("date", ""),
        sender=row.get("from", ""),
        recipient=row.get("to", ""),
        subject=row.get("subject", ""),
        body=row.get("email", ""),
    )


def build_user_prompt(email: EmailRecord) -> str:
    """Return only the raw email body for the downstream model."""
    body = email.body.strip()
    recipient = email.recipient.strip()
    return f"Email to {recipient}:\n\n{body}"


def call_prompt_generator(client: OpenAI, user_prompt: str) -> PromptExtraction:
    """Invoke GPT-4o mini using the structured responses API."""
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content":
                    "Extract the key objectives and facts from the provided email while disregarding the original author's writing style. Then craft a single instruction prompt that begins with 'Write an email' so another LLM can draft a new message achieving the same goal.",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        text_format=PromptExtraction,
    )
    payload: PromptExtraction = response.output_parsed
    normalized_prompt = payload.prompt.strip()
    if not normalized_prompt.lower().startswith("write an email"):
        normalized_prompt = "Write an email " + normalized_prompt.lstrip()
    payload.prompt = normalized_prompt
    return payload


def call_email_writer(client: OpenAI, instruction_prompt: str) -> str:
    """Use GPT-4o mini to draft a style-agnostic email from the generated instruction prompt."""
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=instruction_prompt,
        text_format=DraftEmail,
    )
    return response.output_parsed.email.strip()


def count_rows(csv_path: Path) -> int:
    """Return the number of data rows in the CSV (excluding the header)."""
    with csv_path.open(newline="", encoding="utf-8") as infile:
        return sum(1 for _ in csv.DictReader(infile))


def enrich_row(
    idx: int,
    row: Dict[str, str],
    client_pool: _ClientPool,
    dry_run: bool,
    sleep_seconds: float,
) -> Dict[str, str]:
    """Return the output-ready row with synthesized key points and prompt."""

    record = row_to_record(row)
    prompt = ""
    key_points: list[str] | str = []
    drafted_email = ""

    if dry_run:
        key_points = [
            f"(dry run) placeholder summary for email #{idx}",
        ]
        prompt = (
            "Write an email mirroring the intent of the original message. "
            "This is a dry-run placeholder output."
        )
        drafted_email = (
            "(dry run) This placeholder represents the generated email body."
        )
    else:
        client = client_pool.get()
        if client is None:  # Defensive, should never trigger outside dry-run.
            raise RuntimeError("OpenAI client not available.")

        existing_prompt = row.get("prompt", "")
        existing_key_points = row.get("key_points", "")
        if isinstance(existing_prompt, str) and existing_prompt.strip() and \
           isinstance(existing_key_points, str) and existing_key_points.strip():
            prompt = existing_prompt.strip()
            key_points = json.loads(existing_key_points)
        else:
            payload = call_prompt_generator(client, build_user_prompt(record))
            key_points = payload.key_points
            prompt = payload.prompt + " DO NOT include the subject line."

        existing_email = row.get("llm_email", "")
        if isinstance(existing_email, str) and existing_email.strip():
            drafted_email = existing_email.strip()
        else:
            drafted_email = call_email_writer(client, prompt)

        if sleep_seconds:
            time.sleep(sleep_seconds)

    # Create output row with new columns
    output_row = row.copy()
    output_row["key_points"] = (
        json.dumps(key_points, ensure_ascii=True)
        if not isinstance(key_points, str)
        else key_points
    )
    output_row["prompt"] = prompt
    output_row["llm_email"] = drafted_email

    return output_row


def process_csv(
    csv_path: Path,
    output_path: Path,
    limit: Optional[int],
    sleep_seconds: float,
    dry_run: bool,
    workers: int,
) -> None:
    total_available = count_rows(csv_path)
    total_rows = (
        min(limit, total_available) if limit is not None else total_available
    )
    worker_count = max(1, workers)

    with csv_path.open(newline="", encoding="utf-8") as infile, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        assert_columns(reader.fieldnames or [])

        # Ensure no duplicate columns in output
        base_fieldnames = list(reader.fieldnames)
        for col in ["key_points", "prompt", "llm_email"]:
            if col not in base_fieldnames:
                base_fieldnames.append(col)

        writer = csv.DictWriter(outfile, fieldnames=base_fieldnames)
        writer.writeheader()

        row_iter: Iterable[tuple[int, Dict[str, str]]
                           ] = enumerate(reader, start=1)
        if limit is not None:
            row_iter = islice(row_iter, limit)

        client_pool = _ClientPool(dry_run=dry_run)

        if worker_count <= 1 or total_rows <= 1:
            for idx, row in tqdm(
                row_iter,
                total=total_rows,
                unit="email",
                desc="Processing emails",
            ):
                processed_row = enrich_row(
                    idx=idx,
                    row=row,
                    client_pool=client_pool,
                    dry_run=dry_run,
                    sleep_seconds=sleep_seconds,
                )
                writer.writerow(processed_row)
        else:
            effective_workers = min(worker_count, total_rows or 1)

            def _worker(item: tuple[int, Dict[str, str]]) -> tuple[int, Dict[str, str]]:
                idx, row = item
                processed = enrich_row(
                    idx=idx,
                    row=row,
                    client_pool=client_pool,
                    dry_run=dry_run,
                    sleep_seconds=sleep_seconds,
                )
                return idx, processed

            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                processed_iter = executor.map(_worker, row_iter)
                for _, processed_row in tqdm(
                    processed_iter,
                    total=total_rows,
                    unit="email",
                    desc="Processing emails",
                ):
                    writer.writerow(processed_row)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    csv_path: Path = args.csv_path
    if not csv_path.exists():
        print(f"Input CSV {csv_path} does not exist.", file=sys.stderr)
        return 1

    output_arg = args.output
    if output_arg is None:
        output_path = csv_path.with_suffix(".prompts.csv")
    else:
        if output_arg.suffix.lower() == ".csv":
            output_path = output_arg
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = output_arg
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{csv_path.stem}.prompts.csv"

    try:
        process_csv(
            csv_path=csv_path,
            output_path=output_path,
            limit=args.limit,
            sleep_seconds=args.sleep,
            dry_run=args.dry_run,
            workers=args.workers,
        )
    except Exception as exc:  # Surface helpful context.
        print(f"Failed to process emails: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote augmented prompts to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
