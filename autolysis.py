
# import requests


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# Set your OpenAI API key
# openai.api_key = "your-api-key"
openai.api_key = "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"


def load_csv_files(folder_path):
    """Load all CSV files from the specified folder."""
    dataframes = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            dataframes.append((file, df))
    return dataframes

def analyze_data_with_llm(df):
    """Analyze the dataset using OpenAI API."""
    prompt = f"""
    You are a data analyst. Analyze the following dataset and summarize key insights.
    Dataset preview:
    {df.head(10).to_string(index=False)}
    Columns: {', '.join(df.columns)}
    Please include possible trends, patterns, and outliers in your response.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300
    )
    return response.choices[0].text.strip()

def visualize_data(df, file_name):
    """Create basic visualizations for the dataset."""
    numeric_cols = df.select_dtypes(include="number").columns
    if numeric_cols.empty:
        print(f"No numeric columns to visualize in {file_name}")
        return
    
    plt.figure(figsize=(10, 6))
    sns.pairplot(df[numeric_cols])
    plt.suptitle(f"Visualizations for {file_name}", y=1.02)
    plt.savefig(f"{file_name}_visualization.png")
    plt.close()
    print(f"Visualization saved for {file_name}")

def narrate_story_from_data(file_name, llm_analysis):
    """Create a narrative based on LLM analysis."""
    story_prompt = f"""
    Write a compelling story based on the following dataset analysis for the file '{file_name}':
    {llm_analysis}
    The story should be engaging, easy to understand, and informative.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=story_prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    folder_path = "./"  # Replace with your folder path
    csv_files = load_csv_files(folder_path)

    if not csv_files:
        print("No CSV files found in the folder.")
    else:
        for file_name, df in csv_files:
            print(f"Processing file: {file_name}")

            # Analyze with LLM
            llm_analysis = analyze_data_with_llm(df)
            print(f"Analysis for {file_name}:\n{llm_analysis}\n")

            # Visualize the data
            visualize_data(df, file_name)

            # Narrate a story
            story = narrate_story_from_data(file_name, llm_analysis)
            print(f"Narrative for {file_name}:\n{story}\n")
