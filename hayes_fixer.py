import ast
import csv
import json
import logging
from typing import Any, Dict

def transform_data(data: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
    """
    Transforms a dictionary of phonemes to a dictionary of feature-value pairs.

    Args:
        data (Dict[str, Any]): Original dictionary mapping phonemes to feature lists.

    Returns:
        Dict[str, Dict[str, bool]]: Transformed dictionary with features as keys and boolean values.
    """
    transformed = {}
    for phoneme, feature_list in data.items():
        feature_dict = {}
        for feature in feature_list:
            feature_name = feature[1:].strip()
            feature_value = feature.startswith('+')
            feature_dict[feature_name] = feature_value
        transformed[phoneme] = feature_dict
    return transformed

def extract_and_transform_dicts(script_path: str) -> Dict[str, Any]:
    """
    Extracts dictionaries from a Python script and transforms them.

    Args:
        script_path (str): Path to the Python script.

    Returns:
        Dict[str, Any]: A dictionary containing transformed data for each extracted variable.
    """
    try:
        with open(script_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError as e:
        logging.error(f"File not found: {script_path}")
        raise e

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        logging.error(f"Syntax error in the script: {e}")
        raise e

    optimized_data = {}

    # Iterate through all nodes in the AST
    for node in ast.walk(tree):
        # Check for assignments
        if isinstance(node, ast.Assign):
            # Only process if the value is a dictionary
            if isinstance(node.value, ast.Dict):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        try:
                            original_dict = ast.literal_eval(node.value)
                            transformed_dict = transform_data(original_dict)
                            optimized_data[var_name] = transformed_dict
                            logging.info(f"Processed variable: {var_name}")
                        except Exception as e:
                            logging.warning(f"Skipping variable '{var_name}': {e}")
            else:
                logging.debug("Assignment is not a dictionary; skipping.")

    return optimized_data

def json_to_csv(data: Dict[str, Any], csv_path: str):
    """
    Converts a dictionary containing phoneme features into a CSV file.
    Each phoneme is a column, and each feature is a row.
    This function combines all datasets in the data, ignoring dataset names.

    Args:
        data (Dict[str, Any]): The data containing phoneme features.
        csv_path (str): The output path for the CSV file.
    """
    # Combine all phoneme data from all datasets
    combined_phoneme_data = {}
    for _, phoneme_data in data.items():
        for phoneme, features in phoneme_data.items():
            if phoneme not in combined_phoneme_data:
                combined_phoneme_data[phoneme] = features.copy()
            else:
                # Merge features; existing features take precedence
                for key, value in features.items():
                    if key not in combined_phoneme_data[phoneme]:
                        combined_phoneme_data[phoneme][key] = value

    # Collect all unique features across all phonemes
    all_features = sorted({feature for features in combined_phoneme_data.values() for feature in features})

    # Prepare the CSV file path
    dataset_csv_path = f"{csv_path}.csv"

    # Write to CSV
    with open(dataset_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row: first cell 'Feature', then sorted phoneme symbols
        header = ['Feature'] + sorted(combined_phoneme_data.keys())
        writer.writerow(header)

        # Write each feature row
        for feature in all_features:
            row = [feature]
            for phoneme in sorted(combined_phoneme_data.keys()):
                # Get the feature value for this phoneme, default to None if not present
                value = combined_phoneme_data[phoneme].get(feature, None)
                row.append(value)
            writer.writerow(row)

    logging.info(f"CSV file created: {dataset_csv_path}")

def main():
    """
    Main function to execute the script.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # Specify the input script path and output base path here
    script_path = "hayes2009.py"  # Path to your hayes2009.py file
    output_base_path = "hayes2009"  # Base path for output files (without extension)

    try:
        # Step 1: Extract and transform data from the Python script
        optimized_data = extract_and_transform_dicts(script_path)
        if not optimized_data:
            logging.warning("No valid dictionaries were found and transformed.")
            return

        # Step 2: Save the optimized data to a JSON file
        json_output_path = f"{output_base_path}.json"
        with open(json_output_path, "w", encoding='utf-8') as json_file:
            json.dump(optimized_data, json_file, indent=4, ensure_ascii=False)
        logging.info(f"JSON file created: {json_output_path}")

        # Step 3: Convert the transformed data to CSV
        json_to_csv(optimized_data, output_base_path)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
