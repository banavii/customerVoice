import pandas as pd
df=pd.read_csv("data/processed/cleaned_reviews.csv")
print("dataset shape:",df.shape)
print("/n sample reviews:")
for review in df["review_text"].head(5):
    print("- ",review)
print("\n rating distribution:")
print(df["rating"].value_counts().sort_index())
print("\n product categories:")
print(df["product_category"].value_counts())
print("\n misisng values:")
print(df.isnull().sum().sum())    