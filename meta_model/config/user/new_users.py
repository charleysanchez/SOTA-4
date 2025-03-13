import os
import yaml

# Define the path to your directory
directory_path = "../../temp_data/"
yaml_file_path = "many_users.yaml"

# Load the existing YAML file
with open(yaml_file_path, "r") as file:
    yaml_data = yaml.safe_load(file)

# Get the list of existing sessions to avoid duplicates
existing_sessions = {entry["session"] for entry in yaml_data["dataset"]["test"]}

# Scan the directory and extract session names from filenames
new_entries = []
for filename in os.listdir(directory_path):
    if filename.endswith(".hdf5") or filename.endswith(".json"):  # Adjust for your file types
        session_name = filename.replace(".hdf5", "").replace(".json", "")  # Adjust accordingly
        if session_name not in existing_sessions:
            new_entries.append({"user": 14989984, "session": session_name})

# Append new entries to the dataset
yaml_data["dataset"]["test"].extend(new_entries)

# Save the updated YAML file
with open(yaml_file_path, "w") as file:
    yaml.dump(yaml_data, file, default_flow_style=False)

print(f"Added {len(new_entries)} new entries to the YAML file.")
