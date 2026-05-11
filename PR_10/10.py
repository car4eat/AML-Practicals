# Case Study: Market Basket Analysis


import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data = pd.read_csv("Online Retail.csv")

print(data.head())

# EDA
print("\nMissing Values:")
print(data.isnull().sum())

print("\nTop Products:")
print(data["Description"].value_counts().head())

# Data Cleaning
data.dropna(inplace=True)

data = data[data["Quantity"] > 0]

# Basket Preparation
basket = data.groupby(
    ["InvoiceNo", "Description"]
)["Quantity"].sum().unstack().fillna(0)

basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Apriori Algorithm
frequent_items = apriori(
    basket,
    min_support=0.02,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_items.head())

# Association Rules
rules = association_rules(
    frequent_items,
    metric="confidence",
    min_threshold=0.3
)

print("\nAssociation Rules:")
print(rules[[
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift"
]])

print("\nInterpretation:")
print("Products with high lift are frequently bought together.")
print("These products can be used for recommendations and cross-selling.")