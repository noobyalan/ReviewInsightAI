
from datasets import load_dataset
import pandas as pd


print("loading Amazon Polarity dataset...")
dataset = load_dataset("amazon_polarity")

# check the basic info of the dataset
print("dataset structure:")
print(dataset)

# extract the data
print("extracting the data...")
train_data = dataset["train"]

# check sample
print("sample data:")
first_sample = train_data[0]
print("First Sample:", first_sample)

for i in range(3):
    print(f"Sample {i+1}:")
    print("Label:", "Positive" if train_data[i]['label'] == 1 else "Negative")
    print("Title:", train_data[i]['title'])
    print("Content:", train_data[i]['content'])
    print("-" * 50)

# limit the data quantity
print("筛选样例...")
subset_data = dataset["train"].shuffle(seed=42).select(range(300))

print("转换数据为 Pandas DataFrame...")
data_df = pd.DataFrame(subset_data)

# check the dataset size
print("datasize:", data_df.shape)

# data cleaning
print("开始数据清理...")

# delete the empty comment
data_df = data_df.dropna(subset=["content"])

# restrict the length of the comment 
def truncate_text(text, max_length=512):
    return text[:max_length]

data_df["content"] = data_df["content"].apply(lambda x: truncate_text(x, max_length=512))

data_df["text"] = data_df["title"] + " " + data_df["content"]

print("sample:")
print(data_df.head())

print("保存清理后的数据到 CSV 文件...")
data_df.to_csv("cleaned_amazon_reviews.csv", index=False)

print("数据保存成功！文件名: cleaned_amazon_reviews.csv")