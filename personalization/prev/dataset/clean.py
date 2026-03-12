import json

# Read the dataset
with open('dataset/v5-gpt-5-mini-update-v2/dataset.json', 'r') as f:
    data = json.load(f)

# Fields to remove
fields_to_remove = ['prompt', 'style_agnostic',
                    'task_instruction', 'task_details']

# Remove specified fields from each entry
for entry in data:
    for field in fields_to_remove:
        entry.pop(field, None)

# Write the cleaned dataset back
with open('dataset/v5-gpt-5-mini-update-v2/dataset.json', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Removed fields {fields_to_remove} from {len(data)} entries")
