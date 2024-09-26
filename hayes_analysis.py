import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Set a consistent style for plots
sns.set_style("whitegrid")
sns.set_context("talk")

# Define common palettes and font sizes
CLUSTER_PALETTE = sns.color_palette('Set2', n_colors=3)
HEATMAP_CMAP = "vlag"
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 14
TICKS_FONTSIZE = 12

def load_json(json_path):
    """
    Load the JSON data from a file.

    Parameters:
        json_path (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON data.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def json_to_dataframe(json_data):
    """
    Convert JSON data to a pandas DataFrame.

    Parameters:
        json_data (dict): The JSON data.

    Returns:
        pd.DataFrame: The resulting DataFrame.
    """
    phoneme_data = []
    for dataset in json_data.values():
        for phoneme, features in dataset.items():
            row = {"phoneme": phoneme}
            row.update(features)
            phoneme_data.append(row)
    df = pd.DataFrame(phoneme_data)
    return df

def plot_feature_distribution(df):
    """
    Plot the distribution of phonetic features across phonemes.

    Parameters:
        df (pd.DataFrame): The DataFrame containing phoneme features.
    """
    feature_cols = df.columns.drop('phoneme')
    feature_sums = df[feature_cols].sum().sort_values()

    plt.figure(figsize=(14, 12))
    sns.barplot(
        x=feature_sums.values,
        y=feature_sums.index,
        color='skyblue'
    )
    plt.title("Distribution of Phonetic Features Across Phonemes", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Number of Phonemes with Feature", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Feature", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.tight_layout()
    plt.show()

def perform_pca_with_imputation(df):
    """
    Perform PCA after imputing missing values and encoding categorical variables.

    Parameters:
        df (pd.DataFrame): The DataFrame containing phoneme features.

    Returns:
        pd.DataFrame: The DataFrame with cluster labels added.
    """
    # Select feature columns (exclude 'phoneme')
    feature_cols = df.columns.drop('phoneme')

    # Separate numerical and categorical columns
    numeric_cols = df[feature_cols].select_dtypes(include=['number', 'bool']).columns
    categorical_cols = df[feature_cols].select_dtypes(exclude=['number', 'bool']).columns

    # Convert boolean columns to integers
    df_numeric = df[numeric_cols].copy()
    bool_cols = df_numeric.select_dtypes(include=['bool']).columns
    df_numeric[bool_cols] = df_numeric[bool_cols].astype(int)

    # Impute missing values for numerical columns with 0
    num_imputer = SimpleImputer(strategy='constant', fill_value=0)
    imputed_numeric_data = num_imputer.fit_transform(df_numeric)

    # Impute missing values for categorical columns with 'missing'
    cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
    imputed_categorical_data = cat_imputer.fit_transform(df[categorical_cols])

    # One-hot encode categorical variables
    encoded_categorical_data = pd.get_dummies(
        pd.DataFrame(imputed_categorical_data, columns=categorical_cols)
    )

    # Combine numeric and encoded categorical data
    imputed_data = pd.concat([
        pd.DataFrame(imputed_numeric_data, columns=numeric_cols).reset_index(drop=True),
        encoded_categorical_data.reset_index(drop=True)
    ], axis=1)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(pca_data)

    # Add clusters to the original dataframe
    df['cluster'] = clusters

    # Plot PCA with clusters
    plt.figure(figsize=(14, 12))
    sns.scatterplot(
        x=pca_data[:, 0],
        y=pca_data[:, 1],
        hue=clusters,
        palette=CLUSTER_PALETTE,
        s=150,
        edgecolor='k',
        linewidth=0.5
    )
    plt.title("Phonemes Clustered Based on Features (PCA with Imputed Data)", fontsize=TITLE_FONTSIZE)
    plt.xlabel("PCA Component 1", fontsize=LABEL_FONTSIZE)
    plt.ylabel("PCA Component 2", fontsize=LABEL_FONTSIZE)
    plt.legend(title="Cluster", fontsize=TICKS_FONTSIZE, title_fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.tight_layout()
    plt.show()

    return df

def analyze_feature_by_cluster(df):
    """
    Analyze and plot feature prominence by cluster using a heatmap.

    Parameters:
        df (pd.DataFrame): The DataFrame containing phoneme features and cluster labels.
    """
    # Only keep numeric and boolean columns (exclude 'cluster') for mean calculation
    numeric_cols = df.select_dtypes(include=['number', 'bool']).columns.drop('cluster')
    cluster_feature_means = df.groupby('cluster')[numeric_cols].mean()

    # If you have any NaN values, fill them with zeros
    cluster_feature_means = cluster_feature_means.fillna(0)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cluster_feature_means.T,
        cmap=HEATMAP_CMAP,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={'shrink': 0.5}
    )
    plt.title("Feature Prominence by Cluster", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Cluster", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Phonetic Feature", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE, rotation=0)
    plt.tight_layout()
    plt.show()

def main(json_path):
    """
    Main function to run the phoneme feature analysis.

    Parameters:
        json_path (str): The path to the JSON file containing phoneme data.
    """
    # Load data
    data = load_json(json_path)
    if data is None:
        return

    # Convert to DataFrame
    df = json_to_dataframe(data)

    # Plot feature distribution
    plot_feature_distribution(df)

    # Perform PCA and clustering
    df = perform_pca_with_imputation(df)

    # Analyze and plot feature prominence by cluster
    analyze_feature_by_cluster(df)

if __name__ == "__main__":
    # Provide the path to your JSON file here
    JSON_PATH = "hayes2009.json"

    # Run the analysis
    main(JSON_PATH)