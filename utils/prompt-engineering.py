from typing import Dict, List, Optional


def create_report_prompt(
    findings: Dict[str, int],
    metadata: Dict[str, str],
    template_style: str = 'detailed'
) -> str:
    """
    Create a prompt for generating radiology reports.
    
    Args:
        findings: Dictionary mapping disease names to labels (0, 1, or -1 for uncertain)
        metadata: Patient metadata (sex, age, view, projection)
        template_style: Style of prompt template ('detailed', 'concise', 'structured')
        
    Returns:
        Formatted prompt string
    """
    # Extract metadata
    sex = metadata.get('sex', 'unknown')
    age = metadata.get('age', 'unknown')
    view = metadata.get('view', 'frontal')
    projection = metadata.get('projection', 'PA')
    
    # Categorize findings
    confirmed = []
    possible = []
    uncertain = []
    negative = []
    
    for finding, value in findings.items():
        if finding == "No Finding":
            continue
        
        if value == 1:
            confirmed.append(finding)
        elif value == 0:
            negative.append(finding)
        elif value == -1:
            uncertain.append(finding)
    
    if template_style == 'detailed':
        prompt = f"""
Patient Profile:
- Sex: {sex}
- Age: {age} years
- View: {view}
- Projection: {projection}

Generate a comprehensive radiology report based on the following findings:

Confirmed Findings: {', '.join(confirmed) if confirmed else 'None'}
Ruled Out Findings: {', '.join(negative) if negative else 'None'}
Uncertain Findings: {', '.join(uncertain) if uncertain else 'None'}

Instructions:
1. Use formal medical terminology appropriate for a radiology report.
2. Structure the report with:
   - TECHNIQUE: Brief description of the imaging technique
   - COMPARISON: Note if prior studies available (assume none)
   - FINDINGS: Detailed description of each finding
   - IMPRESSION: Summary of key findings and recommendations
3. For confirmed findings, describe typical radiographic appearances.
4. For uncertain findings, use phrases like "possible," "cannot exclude," or "equivocal for."
5. Maintain professional, objective tone throughout.
6. Include relevant differential diagnoses where appropriate.
7. Suggest follow-up imaging or clinical correlation as needed.

Generate the complete radiology report:
"""
    
    elif template_style == 'concise':
        prompt = f"""
Generate a concise chest X-ray report for a {age}-year-old {sex} patient.
View: {view} {projection}

Positive findings: {', '.join(confirmed) or 'None'}
Equivocal findings: {', '.join(uncertain) or 'None'}

Write a brief report (3-5 sentences) describing the findings and clinical impression.
"""
    
    elif template_style == 'structured':
        prompt = f"""
CHEST X-RAY REPORT TEMPLATE

CLINICAL INFORMATION:
Patient: {age}y {sex}
Technique: {view} {projection} chest radiograph

FINDINGS:
Please generate findings for the following conditions:
- Confirmed abnormalities: {', '.join(confirmed) or 'No confirmed abnormalities'}
- Possible abnormalities: {', '.join(uncertain) or 'None'}
- Normal structures: Describe normal cardiomediastinal silhouette, lungs, pleura, and osseous structures

IMPRESSION:
[Generate 2-3 sentence summary with key findings and recommendations]
"""
    
    else:
        # Default simple prompt
        prompt = f"""
Generate a radiology report for a chest X-ray showing: {', '.join(confirmed) if confirmed else 'no significant abnormalities'}.
Patient is {age} years old, {sex}.
"""
    
    return prompt.strip()


def create_disease_prompts(
    disease_names: List[str],
    style: str = 'descriptive',
    include_negatives: bool = True
) -> Dict[str, str]:
    """
    Create text prompts for each disease category.
    
    Args:
        disease_names: List of disease names
        style: Prompt style ('descriptive', 'clinical', 'simple')
        include_negatives: Whether to include negative prompts
        
    Returns:
        Dictionary mapping disease names to prompts
    """
    prompts = {}
    
    for disease in disease_names:
        if style == 'descriptive':
            # Detailed radiographic descriptions
            disease_descriptions = {
                'Cardiomegaly': 'Chest X-ray showing enlarged cardiac silhouette with cardiothoracic ratio greater than 0.5',
                'Edema': 'Bilateral perihilar haziness with Kerley B lines and vascular redistribution consistent with pulmonary edema',
                'Consolidation': 'Dense opacification with air bronchograms indicating alveolar consolidation',
                'Pneumonia': 'Focal airspace opacity with clinical correlation for infectious pneumonia',
                'Atelectasis': 'Volume loss with displacement of fissures and mediastinal shift indicating atelectasis',
                'Pneumothorax': 'Visible visceral pleural line with absent lung markings peripherally indicating pneumothorax',
                'Pleural Effusion': 'Blunting of costophrenic angle with meniscus sign suggesting pleural fluid',
                'Lung Opacity': 'Nonspecific increased opacity in the lung field',
                'Lung Lesion': 'Discrete nodular or mass-like opacity within the lung parenchyma',
                'Fracture': 'Discontinuity of bone cortex with or without displacement',
                'Support Devices': 'Medical devices including endotracheal tubes, central lines, or pacemakers',
                'Enlarged Cardiomediastinum': 'Widened mediastinal silhouette measuring greater than 8cm on PA view',
                'No Finding': 'Normal chest radiograph with clear lung fields and normal cardiomediastinal silhouette',
                'Pleural Other': 'Pleural abnormality including thickening, calcification, or mass'
            }
            prompt = disease_descriptions.get(
                disease,
                f'Chest X-ray showing evidence of {disease.lower()}'
            )
        
        elif style == 'clinical':
            # Clinical context prompts
            prompt = f'Radiographic findings consistent with clinical diagnosis of {disease.lower()}'
        
        elif style == 'simple':
            # Simple prompts
            prompt = f'{disease} present on chest X-ray'
        
        else:
            prompt = disease
        
        prompts[disease] = prompt
        
        # Add negative prompt if requested
        if include_negatives:
            if disease == 'No Finding':
                prompts[f'Not_{disease}'] = 'Chest X-ray showing multiple abnormalities'
            else:
                prompts[f'Not_{disease}'] = f'Chest X-ray with no evidence of {disease.lower()}'
    
    return prompts


def augment_report_with_findings(
    base_report: str,
    additional_findings: List[str],
    style: str = 'append'
) -> str:
    """
    Augment a base report with additional findings.
    
    Args:
        base_report: Original report text
        additional_findings: List of additional findings to include
        style: How to integrate findings ('append', 'integrate', 'replace')
        
    Returns:
        Augmented report
    """
    if not additional_findings:
        return base_report
    
    if style == 'append':
        # Simply append findings
        addition = '\n\nAdditional findings: ' + ', '.join(additional_findings) + '.'
        return base_report + addition
    
    elif style == 'integrate':
        # Try to integrate into existing structure
        if 'IMPRESSION:' in base_report:
            parts = base_report.split('IMPRESSION:')
            findings_text = ' Also noted: ' + ', '.join(additional_findings) + '.'
            return parts[0] + findings_text + '\nIMPRESSION:' + parts[1]
        else:
            return augment_report_with_findings(base_report, additional_findings, 'append')
    
    elif style == 'replace':
        # Replace findings section
        findings_text = 'FINDINGS: ' + ', '.join(additional_findings)
        if 'FINDINGS:' in base_report:
            import re
            pattern = r'FINDINGS:.*?(?=IMPRESSION:|$)'
            return re.sub(pattern, findings_text + '\n\n', base_report, flags=re.DOTALL)
        else:
            return findings_text + '\n\n' + base_report
    
    return base_report


def create_zero_shot_prompts(
    disease_name: str,
    num_variations: int = 5
) -> List[str]:
    """
    Create multiple prompt variations for zero-shot learning.
    
    Args:
        disease_name: Name of the disease
        num_variations: Number of prompt variations to generate
        
    Returns:
        List of prompt variations
    """
    templates = [
        f"Chest radiograph demonstrating {disease_name}",
        f"X-ray image showing signs of {disease_name}",
        f"Radiological evidence of {disease_name} on chest X-ray",
        f"Chest X-ray with findings consistent with {disease_name}",
        f"Imaging reveals {disease_name}",
        f"PA chest radiograph positive for {disease_name}",
        f"Abnormal chest X-ray indicating {disease_name}",
        f"Radiographic manifestations of {disease_name}",
        f"Chest imaging suspicious for {disease_name}",
        f"X-ray findings suggestive of {disease_name}"
    ]
    
    # Select requested number of variations
    import random
    selected = random.sample(templates, min(num_variations, len(templates)))
    
    # Add some variety with synonyms
    variations = []
    for template in selected:
        # Occasionally replace "chest X-ray" with synonyms
        if random.random() > 0.5:
            template = template.replace("chest X-ray", "chest radiograph")
        elif random.random() > 0.5:
            template = template.replace("X-ray", "radiograph")
        
        variations.append(template)
    
    return variations


def create_comparative_prompts(
    disease1: str,
    disease2: str,
    relationship: str = 'versus'
) -> str:
    """
    Create prompts comparing two diseases.
    
    Args:
        disease1: First disease
        disease2: Second disease  
        relationship: Type of comparison ('versus', 'with', 'progressed_to')
        
    Returns:
        Comparative prompt
    """
    if relationship == 'versus':
        return f"Chest X-ray showing {disease1} versus {disease2}, requiring clinical correlation"
    elif relationship == 'with':
        return f"Chest X-ray demonstrating {disease1} with superimposed {disease2}"
    elif relationship == 'progressed_to':
        return f"Follow-up chest X-ray showing progression from {disease1} to {disease2}"
    else:
        return f"Chest X-ray with findings of both {disease1} and {disease2}"