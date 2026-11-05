import pandas as pd
import os

INPUT_FILE="data/raw/clothing_reviews.csv"
OUTPUT_FILE="data/processed/cleaned_reviews.csv"

def preprocess_reviews():
    print("starting customer preprocessing")
    df=pd.read_csv(INPUT_FILE)
    print("original dataset shape: ",df.shape)

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"],inplace=True)

    df.rename(columns={
        "Clothing ID": "product_id",
        "Age": "customer_age",
        "Title": "review_title",
        "Review Text": "review_text",
        "Rating": "rating",
        "Recommended IND": "recommended",
        "Positive Feedback Count": "helpful_votes",
        "Division Name": "division",
        "Department Name": "department",
        "Class Name": "product_category"
    },inplace=True)

    df.dropna(subset=["review_text"],inplace=True)

    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"] != ""]

    # 7. Clean review text
    df["review_text"] = (
        df["review_text"]
        .str.replace(r"\s+", " ", regex=True)
    )

    # 8. Handle missing optional fields
    df["review_title"] = df["review_title"].fillna("No Title")

    for column in ["division", "department", "product_category"]:
        df[column] = df[column].fillna("Unknown")

    # 9. Remove duplicate records
    df.drop_duplicates(
        subset=["product_id", "review_text", "rating"],
        inplace=True
    )

    # 10. Validate ratings
    df = df[df["rating"].between(1, 5)]

    # 11. Reset index
    df.reset_index(drop=True, inplace=True)

    # 12. Create output directory
    os.makedirs("data/processed", exist_ok=True)

    # 13. Save processed dataset
    df.to_csv(OUTPUT_FILE, index=False)

    # 14. Display results
    print("\n--- PREPROCESSING COMPLETED ---")
    print("Final dataset shape:", df.shape)
    print("Remaining missing values:")
    print(df.isnull().sum())

    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index())

    print("\nSaved cleaned dataset to:", OUTPUT_FILE)


if __name__ == "__main__":
    preprocess_reviews()
    
        
    