import os
import json
import random
import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

COMMON_OCCUPATION_TOKENS = ['historian', 'teacher', 'analyst', 'journalist', 'engineer', 'enthusiast', 'scientist', 'researcher', 'designer', 'genealogist', 'player', 'scholar', 'student', 'producer', 'writer', 'planner', 'critic', 'blogger', 'manager', 'professor', 'developer', 'specialist', 'psychologist', 'sociologist', 'geographer', 'artist', 'director', 'biologist', 'consultant', 'curator', 'architect', 'cartographer', 'linguist', 'anthropologist', 'screenwriter', 'agent', 'athletic', 'collector', 'author', 'organizer', 'entomologist', 'geologist', 'horticulturist', 'listener', 'officer', 'educator', 'botanist', 'consumer', 'musician', 'administrator', 'lawyer', 'statistician', 'librarian', 'coordinator', 'instructor', 'archaeologist', 'editor', 'member',
                            'filmmaker', 'worker', 'counselor', 'trainer', 'activist', 'paleontologist', 'strategist', 'photographer', 'ethnic', 'composer', 'investor', 'politician', 'demographer', 'chemist', 'mathematician', 'archivist', 'physicist', 'singer', 'novelist', 'economist', 'communicator', 'passenger', 'musicologist', 'viewer', 'preservationist', 'nutritionist', 'reviewer', 'astronomer', 'actor', 'tourist', 'traveler', 'commuter', 'driver', 'customer', 'neuroscientist', 'gardener', 'scriptwriter', 'diplomatic', 'conservationist', 'farmer', 'practitioner', 'biochemist', 'theologian', 'programmer', 'therapist', 'advisor', 'environmentalist', 'meteorologist', 'songwriter', 'conductor', 'gamer', 'choreographer', 'guitarist', 'humanitarian', 'ornithologist', 'commentator']
# PERSONA_ATTRIBUTES removed - we now only use the seed and let the LLM expand it


def filter_dataset_by_tokens(
    dataset,
    job_tokens,
    used_seed_data
):
    """Filters the dataset and groups entries by job tokens."""
    token_to_entries = {token: [] for token in job_tokens}
    for item in dataset:
        for token in job_tokens:
            if token in item["input persona"] and item["input persona"] not in used_seed_data:
                token_to_entries[token].append(item)
    return token_to_entries


def sample_from_filtered_data(filtered_data, n):
    """Randomly samples one entry for each token from the filtered data."""
    sampled_entries = []
    available_tokens = [token for token,
                        entries in filtered_data.items() if entries]
    sampled_tokens = random.sample(
        available_tokens, min(n, len(available_tokens)))
    for token in sampled_tokens:
        sampled_entries.append(random.choice(filtered_data[token]))
    return sampled_entries


def sample(n, used_seed_data, json_file_path=None):
    try:
        logger.info(f"Sampling {n} personas from dataset")
        logger.debug(f"Used seed data count: {len(used_seed_data)}")

        # Load dataset
        dataset = load_dataset("proj-persona/PersonaHub",
                               "instruction")["train"]
        logger.debug(f"Loaded dataset with {len(dataset)} entries")

        # Filter dataset by job tokens
        filtered_data = filter_dataset_by_tokens(
            dataset, COMMON_OCCUPATION_TOKENS, used_seed_data)

        # Sample entries from filtered data
        sampled_entries = sample_from_filtered_data(filtered_data, n)

        if json_file_path:
            with open(json_file_path, "w") as file:
                json.dump(sampled_entries, file, indent=4)
                logger.debug(f"Saved sampled entries to {json_file_path}")

        logger.info(f"Successfully sampled {len(sampled_entries)} personas")
        return sampled_entries

    except Exception as e:
        logger.error(f"Error sampling personas: {str(e)}")
        logger.exception("Full traceback for persona sampling:")
        raise


class PersonaTemplateSampler:
    def __init__(
        self,
        verbose=False
    ):
        self.verbose = verbose
        try:
            logger.info("Initializing PersonaTemplateSampler")
            self.seed_dataset = load_dataset(
                "proj-persona/PersonaHub", "instruction")['train']
            logger.debug(
                f"Loaded seed dataset with {len(self.seed_dataset)} entries")
            logger.info("Successfully initialized PersonaTemplateSampler")

        except Exception as e:
            logger.error(
                f"Error initializing PersonaTemplateSampler: {str(e)}")
            logger.exception(
                "Full traceback for PersonaTemplateSampler initialization:")
            raise

    def __call__(self, n, used_seed_data, random_seed=42):
        try:
            logger.info(f"Generating {n} persona templates (seeds only)")
            logger.debug(f"Random seed: {random_seed}")

            random.seed(random_seed)
            dataset = []
            n_remaining = n

            while n_remaining > 0:
                logger.debug(f"Need {n_remaining} more personas")
                new_personas = sample(n_remaining, used_seed_data)
                dataset += new_personas
                used_seed_data += [item['input persona']
                                   for item in new_personas]
                n_remaining = n - len(dataset)

            # Only keep the seed - LLM will expand all other attributes
            for i, item in enumerate(dataset):
                persona = {"id": i, "seed": item['input persona']}
                dataset[i] = persona

            logger.info(
                f"Successfully generated {len(dataset)} persona templates")
            return dataset

        except Exception as e:
            logger.error(f"Error generating persona templates: {str(e)}")
            logger.exception("Full traceback for persona template generation:")
            raise
