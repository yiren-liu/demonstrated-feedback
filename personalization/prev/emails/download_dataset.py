import csv
import re
import shutil
import sys
from email import policy
from email.parser import Parser
from pathlib import Path
from typing import Tuple

try:
    import kagglehub
except ImportError:  # Falls back to local CSV if kagglehub is unavailable.
    kagglehub = None

csv.field_size_limit(sys.maxsize)

SENT_FOLDERS = {"_sent", "_sent_mail", "sent", "sent_items"}
X_ORIGIN_RE = re.compile(r"^X-Origin:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
FORWARD_PATTERNS = [
    re.compile(r"^Subject:\s*(FW:|FWD:|Fwd:)", re.IGNORECASE | re.MULTILINE),
    re.compile(r"-----Original Message-----", re.IGNORECASE),
    re.compile(r"Forwarded by", re.IGNORECASE),
    re.compile(r"---------- Forwarded message ---------", re.IGNORECASE),
]
MIN_BODY_WORDS = 5
OUTPUT_HEADER = ["file", "date", "from", "to", "subject", "email"]
EMAIL_PARSER = Parser(policy=policy.default)


def slugify(value: str) -> str:
    """Return a canonical slug for employee comparisons."""
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def is_sent_by_employee(file_path: str, message: str, employee: str) -> bool:
    """Return True if the message belongs to a sent-mail style folder for this employee."""
    parts = file_path.split("/")
    if len(parts) < 2:
        return False

    folder = parts[1].lower()
    if folder not in SENT_FOLDERS:
        return False

    origin_match = X_ORIGIN_RE.search(message)
    if origin_match:
        origin_slug = slugify(origin_match.group(1))
        if origin_slug and origin_slug != slugify(employee):
            return False

    return True


def is_forwarded_email(message: str) -> bool:
    """Return True if the message looks like a forwarded email."""
    return any(pattern.search(message) for pattern in FORWARD_PATTERNS)


def body_has_min_words(body: str, minimum: int = MIN_BODY_WORDS) -> bool:
    """Return True when the body contains at least `minimum` word tokens."""
    if not body:
        return False
    words = re.findall(r"\b\w+\b", body)
    return len(words) >= minimum


def extract_email_fields(message: str) -> Tuple[str, str, str, str, str]:
    """Extract Date, From, To, Subject, and body text from the email content."""
    if not message:
        return "", "", "", "", ""

    normalized = message.replace("\r\n", "\n")
    date_val = from_addr = to_addr = subject = ""
    body = ""

    try:
        email_obj = EMAIL_PARSER.parsestr(normalized)
    except Exception:
        email_obj = None

    if email_obj is not None:
        date_val = email_obj.get("Date", "") or ""
        from_addr = email_obj.get("From", "") or ""
        to_addr = email_obj.get("To", "") or ""
        subject = email_obj.get("Subject", "") or ""

        if email_obj.is_multipart():
            parts = []
            for part in email_obj.walk():
                if part.get_content_maintype() == "multipart":
                    continue
                if part.get_content_disposition() == "attachment":
                    continue
                try:
                    parts.append(part.get_content())
                except Exception:
                    payload = part.get_payload(decode=True)
                    if payload is None:
                        continue
                    charset = part.get_content_charset() or "utf-8"
                    parts.append(payload.decode(charset, errors="replace"))
            body = "\n".join(part for part in parts if part)
        else:
            try:
                body = email_obj.get_content()
            except Exception:
                payload = email_obj.get_payload(decode=True)
                if payload is None:
                    body = email_obj.get_payload()
                else:
                    charset = email_obj.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="replace")

    if not from_addr:
        match = re.search(r"^From:[ \t]*(.+)$", normalized, re.MULTILINE)
        if match:
            from_addr = match.group(1).strip()
    if not to_addr:
        match = re.search(r"^To:[ \t]*(.+)$", normalized, re.MULTILINE)
        if match:
            to_addr = match.group(1).strip()
    if not subject:
        match = re.search(r"^Subject:[ \t]*(.*)$", normalized, re.MULTILINE)
        if match:
            subject = match.group(1).strip()
    if not date_val:
        match = re.search(r"^Date:[ \t]*(.+)$", normalized, re.MULTILINE)
        if match:
            date_val = match.group(1).strip()

    if not body:
        parts = normalized.split("\n\n", 1)
        body = parts[1] if len(parts) == 2 else normalized

    return (
        date_val.strip(),
        from_addr.strip(),
        to_addr.strip(),
        subject.strip(),
        body.strip(),
    )


def download_dataset() -> Path:
    """Download the Enron email dataset via kagglehub."""
    if kagglehub is None:
        raise ModuleNotFoundError(
            "kagglehub is not installed. Install it or place emails.csv next to this script."
        )
    path = Path(kagglehub.dataset_download("wcukierski/enron-email-dataset"))
    print("Path to dataset files:", path)
    return path


def split_emails_by_employee(source_csv: Path, output_dir: Path) -> None:
    """Split emails.csv into one CSV per employee based on the first path segment."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    files = {}

    try:
        with source_csv.open(newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"No header row found in {source_csv}")

            for row in reader:
                if not row:
                    continue

                file_path = row[0]
                employee = file_path.split("/", 1)[0]
                if not employee:
                    employee = "unknown"

                message = row[1] if len(row) > 1 else ""
                if not is_sent_by_employee(file_path, message, employee):
                    continue
                if is_forwarded_email(message):
                    continue

                date_val, from_addr, to_addr, subject, email_body = extract_email_fields(
                    message)
                if not body_has_min_words(email_body):
                    continue

                writer = writers.get(employee)
                if writer is None:
                    out_path = output_dir / f"{employee}.csv"
                    outfile = out_path.open("w", newline="", encoding="utf-8")
                    files[employee] = outfile
                    writer = csv.writer(outfile)
                    writer.writerow(OUTPUT_HEADER)
                    writers[employee] = writer

                writer.writerow(
                    [file_path, date_val, from_addr, to_addr, subject, email_body])
    finally:
        for fh in files.values():
            fh.close()


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent
    local_csv = dataset_dir / "emails.csv"

    if local_csv.exists():
        print(f"Using local dataset at {local_csv}")
        source_csv = local_csv
    else:
        downloaded_path = download_dataset()
        source_csv = downloaded_path / "emails.csv"
        if not source_csv.exists():
            raise FileNotFoundError(
                f"emails.csv not found in downloaded dataset at {downloaded_path}"
            )

    output_dir = dataset_dir / "emails_by_employee"
    split_emails_by_employee(source_csv, output_dir)
    print(f"Wrote per-employee CSV files to {output_dir}")


if __name__ == "__main__":
    main()
