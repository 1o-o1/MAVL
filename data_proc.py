import pandas as pd
import numpy as np
from pathlib import Path
import ollama

# Configuration
chestxrays_root = Path("path/to/CheXpert-v1.0-small")
train_csv =  "valid.csv"

# Load dataset
full_train_df = pd.read_csv(train_csv)

# Extract patient and study IDs
full_train_df[['patient', 'study']] = full_train_df.Path.str.split('/', expand=True)[[2, 3]]

# Define findings list
findings = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

# Preprocess labels
u_one_features = ['Atelectasis', 'Edema']
u_zero_features = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']

for col in u_one_features:
    full_train_df[col] = full_train_df[col].replace(-1, 1)
for col in u_zero_features:
    full_train_df[col] = full_train_df[col].replace(-1, 0)
full_train_df = full_train_df.replace(-1, np.nan)

def generate_prompt(row):
    # Extract patient metadata
    sex = row['Sex']
    age = row['Age']
    view = row['Frontal/Lateral']
    projection = row['AP/PA']
    
    # Categorize findings
    confirmed = []
    possible = []
    uncertain = []
    
    for finding in findings[1:]:  # Skip "No Finding"
        if pd.isna(row[finding]):
            uncertain.append(finding)
        elif row[finding] == 1:
            confirmed.append(finding)
        elif row[finding] == 0:
            possible.append(finding)
    
    # Construct prompt
    prompt = f"""
    Patient Profile:
    - Sex: {sex}
    - Age: {age} years
    - View: {view}
    - Projection: {projection}

    Generate a concise radiology report based on the following findings:
    Confirmed Findings: {', '.join(confirmed) or 'None'}
    Possible Findings: {', '.join(possible) or 'None'}
    Uncertain Findings: {', '.join(uncertain) or 'None'}

    Instructions:
    1. Use formal medical terminology.
    2. Structure the report with an introduction, findings, and conclusion.
    3. For uncertain findings, mention them as "possible" or "equivocal."
    4. Avoid listing findings; integrate them into sentences.
    5. Example format:
    'This is a chest X-ray of a [sex] patient aged [age]. The study demonstrates [finding1] and [finding2]. Additionally, there is evidence of [finding3]. No other abnormalities were detected.'
    """
    return prompt

def generate_ollama_report(prompt):
    response = ollama.chat(
        model='deepseek-r1:14b',
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.7,
            "num_predict": 256  # Use "num_predict" instead of "max_tokens"
        }
    )
    return response['message']['content']

# Generate reports
full_train_df['report'] = full_train_df.apply(
    lambda row: generate_ollama_report(generate_prompt(row)), axis=1
)


# Save results
output_path =  "train_with_fluent_reports.csv"
full_train_df.to_csv(output_path, index=False)
print(f"Saved enhanced reports to {output_path}")