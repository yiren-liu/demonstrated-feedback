"""
Streamlit app for browsing and comparing genre-holdout generation results.

Usage:
    streamlit run browse_generations.py [--server.port PORT]

To expose via ngrok (for remote access):
    1. Set your ngrok authtoken:  ngrok config add-authtoken YOUR_TOKEN
       or:  NGROK_AUTH_TOKEN=YOUR_TOKEN streamlit run browse_generations.py
    2. Set NGROK=1 to enable:
       NGROK=1 streamlit run browse_generations.py --server.port 8501
"""

import json
import csv
import os
import datetime
from pathlib import Path

import sys

# --- ngrok tunnel (opt-in via NGROK=1 env var) ---
_ngrok_url = None
if os.environ.get("NGROK") == "1":
    from pyngrok import ngrok, conf
    # Only connect once (check if tunnel already exists)
    auth_token = os.environ.get("NGROK_AUTH_TOKEN")
    if auth_token:
        conf.get_default().auth_token = auth_token
    existing = ngrok.get_tunnels()
    if existing:
        _ngrok_url = existing[0].public_url
    else:
        port = int(os.environ.get("STREAMLIT_SERVER_PORT", "8501"))
        _ngrok_url = ngrok.connect(port, "http").public_url
    # Print to stderr so it appears in the terminal
    print(f"\n  ngrok tunnel: {_ngrok_url}\n", file=sys.stderr)

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent / "outputs"

# ---------------------------------------------------------------------------
# Method registry – maps display name → (generation root, eval_results dir)
# ---------------------------------------------------------------------------
METHODS = {
    "DITTO": {
        "gen_root": BASE_DIR / "genre_holdout_exp",
        "eval_dir": BASE_DIR / "genre_holdout_exp" / "eval_results",
    },
    "SFT-Only": {
        "gen_root": BASE_DIR / "genre_holdout_sft_exp",
        "eval_dir": BASE_DIR / "genre_holdout_sft_exp" / "eval_results",
    },
    "RAG (GPT-5.2)": {
        "gen_root": BASE_DIR / "genre_holdout_rag_exp" / "gpt-5.2",
        "eval_dir": BASE_DIR / "genre_holdout_rag_exp" / "gpt-5.2" / "eval_results",
    },
    "Prompt (GPT-5.2)": {
        "gen_root": BASE_DIR / "genre_holdout_prompt_exp" / "gpt-5.2",
        "eval_dir": BASE_DIR / "genre_holdout_prompt_exp" / "gpt-5.2" / "eval_results",
    },
    "Steering (Mistral-7B)": {
        "gen_root": BASE_DIR / "genre_holdout_steering_exp" / "mistral-7b",
        "eval_dir": BASE_DIR / "genre_holdout_steering_exp" / "mistral-7b" / "eval_results",
    },
}

AUTHORS = [f"a{i}" for i in range(10)]
GENRES = ["b", "c", "d", "e", "s"]

NOTES_FILE = Path(__file__).resolve().parent / "case_study_notes.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_generations(gen_root: str, subdir: str):
    path = Path(gen_root) / subdir / "generations.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_style_profile(gen_root: str, subdir: str):
    path = Path(gen_root) / subdir / "style_profile.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_eval_csv(eval_dir: str, filename: str):
    path = Path(eval_dir) / filename
    if path.exists():
        with open(path) as f:
            reader = csv.DictReader(f)
            return list(reader)
    return None


def build_rating_lookup(rows):
    """Build dict: (condition, variant, author_key, prompt_idx, gen_idx) → rating"""
    lookup = {}
    if rows is None:
        return lookup
    for r in rows:
        key = (r["condition"], r["variant"], int(r["author_key"]),
               int(r["prompt_idx"]), int(r["gen_idx"]))
        lookup[key] = int(r["rating"])
    return lookup


def build_pairwise_lookup(rows):
    """Build dict: (author, unseen_variant, prompt_text_prefix, gen_idx) → winner"""
    lookup = {}
    if rows is None:
        return lookup
    for r in rows:
        key = (r["author"], r["unseen_variant"], r["prompt"][:60],
               int(r["gen_idx"]))
        lookup[key] = r["winner"]
    return lookup


def get_condition_variant(subdir: str):
    """Parse 'seen-s0-a3' → ('seen', 's0') or 'unseen-genre-b-a3' → ('unseen', 'genre-b')"""
    if subdir.startswith("seen"):
        return "seen", "s0"
    parts = subdir.split("-")
    # unseen-genre-X-aY
    genre_letter = parts[2]
    return "unseen", f"genre-{genre_letter}"


def load_notes():
    if NOTES_FILE.exists():
        with open(NOTES_FILE) as f:
            return json.load(f)
    return {}


def save_notes(notes):
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f, indent=2)


def rating_badge(rating):
    if rating is None:
        return ""
    if rating >= 5:
        color = "green"
    elif rating >= 3:
        color = "orange"
    else:
        color = "red"
    return f":{color}[Rating: {rating}/7]"


def winner_badge(winner):
    if winner == "a":
        return ":green[Seen wins]"
    elif winner == "b":
        return ":red[Unseen wins]"
    elif winner == "tie":
        return ":gray[Tie]"
    return ""


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Genre Holdout Browser", layout="wide")
st.title("Genre Holdout: Generation Browser")

if _ngrok_url:
    st.sidebar.success(f"ngrok: {_ngrok_url}")

# ---------------------------------------------------------------------------
# Sidebar – global filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

page = st.sidebar.radio("View", ["Single Method", "Cross-Method Compare"], index=0)

author = st.sidebar.selectbox("Author", AUTHORS, index=0)
author_key = int(author[1:])

genre = st.sidebar.selectbox("Unseen Genre Variant", GENRES, index=0)

condition = st.sidebar.radio("Condition", ["seen", "unseen", "side-by-side"], index=2)

# ---------------------------------------------------------------------------
# Notes panel (persisted to JSON)
# ---------------------------------------------------------------------------
notes = load_notes()

# ===========================================================================
# PAGE: Single Method
# ===========================================================================
if page == "Single Method":
    method_name = st.sidebar.selectbox("Method", list(METHODS.keys()))
    cfg = METHODS[method_name]
    gen_root = cfg["gen_root"]
    eval_dir = cfg["eval_dir"]

    # Load eval data
    rating_lookup = {}
    pairwise_lookup = {}
    if eval_dir and Path(eval_dir).exists():
        rating_rows = load_eval_csv(str(eval_dir), "eval_results_rating.csv")
        pairwise_rows = load_eval_csv(str(eval_dir), "eval_results_pairwise.csv")
        rating_lookup = build_rating_lookup(rating_rows)
        pairwise_lookup = build_pairwise_lookup(pairwise_rows)

    # Determine subdirs
    seen_subdir = f"seen-s0-{author}"
    unseen_subdir = f"unseen-genre-{genre}-{author}"

    if condition == "side-by-side":
        subdirs = [("seen", seen_subdir), ("unseen", unseen_subdir)]
    elif condition == "seen":
        subdirs = [("seen", seen_subdir)]
    else:
        subdirs = [("unseen", unseen_subdir)]

    # Load generations
    gen_data = {}
    for label, sd in subdirs:
        data = load_generations(str(gen_root), sd)
        if data:
            gen_data[label] = data

    if not gen_data:
        st.warning("No generation data found for this selection.")
        st.stop()

    # Pick a reference set of prompts from whichever is available
    ref_data = gen_data.get("seen") or gen_data.get("unseen")
    num_prompts = len(ref_data["results"])

    prompt_idx = st.sidebar.number_input("Prompt index", 0, num_prompts - 1, 0)
    gen_idx = st.sidebar.number_input("Generation index (0-2)", 0, 2, 0)

    st.subheader(f"{method_name} | Author {author} | Unseen genre: {genre}")

    # Show style profile for prompt baseline
    if method_name.startswith("Prompt"):
        profile = load_style_profile(str(gen_root), seen_subdir)
        if profile:
            with st.expander("Style Profile (used in prompt)"):
                st.write(profile.get("style_profile", ""))

    # Reference text
    ref_result = ref_data["results"][prompt_idx]
    st.markdown("### Prompt")
    st.info(ref_result["prompt"])

    st.markdown("### Reference (Author's Original)")
    st.text_area("Reference", ref_result["reference"], height=200,
                 disabled=True, key="ref_single")

    # Show generations
    if condition == "side-by-side":
        col_seen, col_unseen = st.columns(2)

        with col_seen:
            st.markdown("#### Seen Condition")
            if "seen" in gen_data:
                seen_result = gen_data["seen"]["results"][prompt_idx]
                r = rating_lookup.get(("seen", "s0", author_key, prompt_idx, gen_idx))
                if r is not None:
                    st.markdown(rating_badge(r))
                gen_text = seen_result["generations"][gen_idx] if gen_idx < len(seen_result["generations"]) else "(no generation)"
                st.text_area("Seen generation", gen_text, height=400,
                             disabled=True, key="seen_gen_single")
            else:
                st.warning("No seen data")

        with col_unseen:
            st.markdown("#### Unseen Condition")
            if "unseen" in gen_data:
                unseen_result = gen_data["unseen"]["results"]
                # prompts may differ between seen/unseen; find matching prompt_idx
                if prompt_idx < len(unseen_result):
                    ur = unseen_result[prompt_idx]
                    r = rating_lookup.get(("unseen", f"genre-{genre}", author_key, prompt_idx, gen_idx))
                    if r is not None:
                        st.markdown(rating_badge(r))
                    gen_text = ur["generations"][gen_idx] if gen_idx < len(ur["generations"]) else "(no generation)"
                    st.text_area("Unseen generation", gen_text, height=400,
                                 disabled=True, key="unseen_gen_single")
                else:
                    st.warning("Prompt index out of range for unseen condition")
            else:
                st.warning("No unseen data")

        # Pairwise result
        pw_key = (author, f"genre-{genre}",
                  ref_result["prompt"][:60], gen_idx)
        pw = pairwise_lookup.get(pw_key)
        if pw:
            st.markdown(f"**Pairwise verdict:** {winner_badge(pw)}")

    else:
        # Single condition view
        label = condition
        data = gen_data[label]
        result = data["results"][prompt_idx]
        cond_str, var_str = get_condition_variant(subdirs[0][1])
        r = rating_lookup.get((cond_str, var_str, author_key, prompt_idx, gen_idx))
        if r is not None:
            st.markdown(rating_badge(r))
        gen_text = result["generations"][gen_idx] if gen_idx < len(result["generations"]) else "(no generation)"
        st.text_area("Generation", gen_text, height=400,
                     disabled=True, key="gen_single_one")

    # Notes
    st.markdown("---")
    st.markdown("### Notes")
    note_key = f"{method_name}|{author}|{genre}|{condition}|p{prompt_idx}|g{gen_idx}"
    existing_note = notes.get(note_key, "")
    new_note = st.text_area("Your observations", existing_note, height=100,
                            key="note_single")
    if st.button("Save Note", key="save_single"):
        if new_note.strip():
            notes[note_key] = new_note.strip()
        elif note_key in notes:
            del notes[note_key]
        save_notes(notes)
        st.success("Note saved!")


# ===========================================================================
# PAGE: Cross-Method Compare
# ===========================================================================
elif page == "Cross-Method Compare":
    selected_methods = st.sidebar.multiselect(
        "Methods to compare",
        list(METHODS.keys()),
        default=list(METHODS.keys())
    )

    if not selected_methods:
        st.warning("Select at least one method.")
        st.stop()

    # Use first method to get prompt list
    first_cfg = METHODS[selected_methods[0]]
    first_subdir = f"seen-s0-{author}"
    first_data = load_generations(str(first_cfg["gen_root"]), first_subdir)
    if not first_data:
        # try unseen
        first_subdir = f"unseen-genre-{genre}-{author}"
        first_data = load_generations(str(first_cfg["gen_root"]), first_subdir)
    if not first_data:
        st.warning("No data found for this author.")
        st.stop()

    num_prompts = len(first_data["results"])
    prompt_idx = st.sidebar.number_input("Prompt index", 0, num_prompts - 1, 0)
    gen_idx = st.sidebar.number_input("Generation index (0-2)", 0, 2, 0)

    st.subheader(f"Cross-Method | Author {author} | Unseen genre: {genre}")

    # Show prompt and reference
    ref_result = first_data["results"][prompt_idx]
    st.markdown("### Prompt")
    st.info(ref_result["prompt"])

    st.markdown("### Reference (Author's Original)")
    st.text_area("Reference", ref_result["reference"], height=200,
                 disabled=True, key="ref_cross")

    # Build columns for each method
    if condition == "side-by-side":
        st.markdown("### Seen vs Unseen by Method")
        for method_name in selected_methods:
            cfg = METHODS[method_name]
            gen_root = cfg["gen_root"]
            eval_dir = cfg["eval_dir"]

            rating_lookup = {}
            if eval_dir and Path(eval_dir).exists():
                rating_rows = load_eval_csv(str(eval_dir), "eval_results_rating.csv")
                rating_lookup = build_rating_lookup(rating_rows)

            seen_data = load_generations(str(gen_root), f"seen-s0-{author}")
            unseen_data = load_generations(str(gen_root), f"unseen-genre-{genre}-{author}")

            st.markdown(f"#### {method_name}")
            col_s, col_u = st.columns(2)

            with col_s:
                st.markdown("**Seen**")
                if seen_data and prompt_idx < len(seen_data["results"]):
                    r = rating_lookup.get(("seen", "s0", author_key, prompt_idx, gen_idx))
                    if r is not None:
                        st.markdown(rating_badge(r))
                    gens = seen_data["results"][prompt_idx]["generations"]
                    text = gens[gen_idx] if gen_idx < len(gens) else "(no generation)"
                    st.text_area("", text, height=300, disabled=True,
                                 key=f"cross_seen_{method_name}")
                else:
                    st.caption("No data")

            with col_u:
                st.markdown("**Unseen**")
                if unseen_data and prompt_idx < len(unseen_data["results"]):
                    r = rating_lookup.get(("unseen", f"genre-{genre}", author_key, prompt_idx, gen_idx))
                    if r is not None:
                        st.markdown(rating_badge(r))
                    gens = unseen_data["results"][prompt_idx]["generations"]
                    text = gens[gen_idx] if gen_idx < len(gens) else "(no generation)"
                    st.text_area("", text, height=300, disabled=True,
                                 key=f"cross_unseen_{method_name}")
                else:
                    st.caption("No data")

            st.markdown("---")

    else:
        # Single condition across methods
        cond_label = condition
        subdir_template = f"seen-s0-{author}" if cond_label == "seen" else f"unseen-genre-{genre}-{author}"

        cols = st.columns(len(selected_methods))
        for i, method_name in enumerate(selected_methods):
            cfg = METHODS[method_name]
            gen_root = cfg["gen_root"]
            eval_dir = cfg["eval_dir"]

            rating_lookup = {}
            if eval_dir and Path(eval_dir).exists():
                rating_rows = load_eval_csv(str(eval_dir), "eval_results_rating.csv")
                rating_lookup = build_rating_lookup(rating_rows)

            data = load_generations(str(gen_root), subdir_template)

            with cols[i]:
                st.markdown(f"#### {method_name}")
                if data and prompt_idx < len(data["results"]):
                    cond_str, var_str = get_condition_variant(subdir_template)
                    r = rating_lookup.get((cond_str, var_str, author_key, prompt_idx, gen_idx))
                    if r is not None:
                        st.markdown(rating_badge(r))
                    gens = data["results"][prompt_idx]["generations"]
                    text = gens[gen_idx] if gen_idx < len(gens) else "(no generation)"
                    st.text_area("", text, height=400, disabled=True,
                                 key=f"cross_single_{method_name}")
                else:
                    st.caption("No data")

    # Notes
    st.markdown("---")
    st.markdown("### Notes")
    note_key = f"cross|{','.join(selected_methods)}|{author}|{genre}|{condition}|p{prompt_idx}|g{gen_idx}"
    existing_note = notes.get(note_key, "")
    new_note = st.text_area("Your observations", existing_note, height=100,
                            key="note_cross")
    if st.button("Save Note", key="save_cross"):
        if new_note.strip():
            notes[note_key] = new_note.strip()
        elif note_key in notes:
            del notes[note_key]
        save_notes(notes)
        st.success("Note saved!")

# ---------------------------------------------------------------------------
# Sidebar: View saved notes
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Saved Notes")
notes = load_notes()
if notes:
    if st.sidebar.button("Show all notes"):
        st.sidebar.json(notes)
    st.sidebar.caption(f"{len(notes)} note(s) saved")

    if st.sidebar.button("Export notes to Markdown"):
        md_lines = [f"# Case Study Notes\n\nExported: {datetime.datetime.now().isoformat()}\n"]
        for k, v in sorted(notes.items()):
            md_lines.append(f"## `{k}`\n\n{v}\n")
        md_path = Path(__file__).resolve().parent / "case_study_notes.md"
        md_path.write_text("\n".join(md_lines))
        st.sidebar.success(f"Exported to {md_path}")
else:
    st.sidebar.caption("No notes yet")
