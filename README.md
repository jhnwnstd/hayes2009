# Hayes Features Data Processing and Analysis

This repository contains scripts for processing the Hayes (2009) phonological features data and performing preliminary analysis. The scripts allow you to:

- Convert the original data from a Python script (`hayes2009.py`) into both JSON and CSV formats.
- Perform preliminary analysis on the data, including visualization of feature distributions and clustering of phonemes based on their features.

## Table of Contents

- [Background](#background)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Conversion](#data-conversion)
  - [Data Analysis](#data-analysis)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Background

Bruce Hayes' (2009) work provides a set of phonological features for various phonemes across different languages. This repository aims to process that data for further computational analysis by converting it into machine-readable formats (JSON and CSV) and performing preliminary exploratory data analysis.

## Features

- **Data Conversion**: Convert the original phoneme feature data from a Python script into JSON and CSV formats.
- **Data Analysis**:
  - Visualize the distribution of phonetic features across phonemes.
  - Perform Principal Component Analysis (PCA) to reduce dimensionality.
  - Cluster phonemes using KMeans clustering.
  - Visualize clusters and analyze feature prominence within clusters.

## Requirements

- **Python**: Version 3.x
- **Python Packages**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

You can install the required packages using:

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**

```plaintext
# Data Manipulation and Analysis
pandas>=1.3.0
numpy>=1.21.0

# Visualization Libraries
matplotlib>=3.4.0
seaborn>=0.11.0

# Machine Learning Libraries
scikit-learn>=0.24.0
```

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/jhnwnstd/hayes2009.git
   cd hayes2009
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Conversion

The script `hayes_converter.py` reads the original `hayes2009.py` file containing phoneme features and converts it into both JSON and CSV formats.

#### Steps:

1. **Place `hayes2009.py`**:

   Ensure that the `hayes2009.py` file is in the same directory as `hayes_converter.py`.

2. **Run the Script**:

   ```bash
   python hayes_converter.py
   ```

3. **Output**:

   - `hayes2009.json`: A JSON file containing the transformed phoneme feature data.
   - `hayes2009.csv`: A CSV file where each phoneme is a column and each feature is a row.

#### Script Explanation:

- **Input**: `hayes2009.py` - A Python script containing dictionaries of phonemes and their features.
- **Processing**:
  - Parses the Python script to extract phoneme data.
  - Transforms the data into a structured format.
  - Saves the data into both JSON and CSV formats.
- **Output**: `hayes2009.json` and `hayes2009.csv`.

### Data Analysis

The script `hayes_analysis.py` performs preliminary analysis on the phoneme feature data.

#### Steps:

1. **Ensure `hayes2009.json` Exists**:

   Confirm that `hayes2009.json` is present in the directory (generated from the previous step).

2. **Run the Analysis Script**:

   ```bash
   python hayes_analysis.py
   ```

3. **Output**:

   The script will display several plots:

   - **Feature Distribution**: A bar plot showing the distribution of phonetic features across phonemes.
   - **PCA Scatter Plot**: A scatter plot of phonemes in reduced dimensionality space, colored by cluster.
   - **Feature Heatmap**: A heatmap showing feature prominence by cluster.

#### Script Explanation:

- **Loading Data**: Reads the JSON file containing phoneme features.
- **Visualization**:
  - **Feature Distribution**: Shows how common each feature is across phonemes.
  - **PCA Scatter Plot**: Displays phonemes in reduced dimensionality space, colored by cluster.
  - **Feature Heatmap**: Illustrates the prominence of each feature within clusters.
- **Clustering**: Groups phonemes based on their features using KMeans clustering.

## Results

By running these scripts, you will obtain:

- **Processed Data**: Phoneme features in JSON and CSV formats for easy manipulation.
- **Insights**:
  - Understanding of which features are most common across phonemes.
  - Visualization of how phonemes cluster based on their features.
  - Analysis of which features are most prominent within each cluster.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Bruce Hayes (2009)**: For providing the original phoneme feature data.
- **Open-Source Libraries**: Thanks to the contributors of `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.