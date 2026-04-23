## Experimental Setup

**Dataset.** We use the CMCC corpus, which contains texts from 10 authors across 5 genres (blogs, chat, discussion, emails, essays) and 6 topics, yielding up to 30 (genre, topic) writing tasks per author.

**Conditions.** We evaluate generalization under three increasingly challenging data splits, each constructed independently per author:

- **Seen**: A random 8-example train / rest-test split (3 seeds).
- **Unseen-Genre**: All examples from one held-out genre are reserved for testing; the model trains on the remaining 4 genres. We repeat this for each of the 5 genres.
- **Unseen-Topic**: All examples from a pair of held-out topics go to test (3 topic-pair variants: {Iraq, Gender Discrimination}, {Catholic Church, Gay Marriage}, {Privacy, Marijuana}).

**Training.** For each condition × author, we train a separate DITTO model on Mistral-7B-Instruct-v0.2 using LoRA (r=32, α=64). Training proceeds in two phases: SFT on author demonstrations (≤30 steps, early-stopped at loss < 1.0), followed by iterative DPO with online comparison generation (40 steps, mixing 70% expert-vs-current, 20% expert-vs-replay, 10% inter-policy pairs).

**Generation & Evaluation.** We generate 3 samples per test prompt from each trained model. A GPT-4o judge rates each generation against the author's ground-truth reference on a 1–7 Likert scale for stylistic similarity, where 1 = extremely different and 7 = extremely similar.
