import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Step 1: 加载数据
print("Loading data...")
data_df = pd.read_csv("amazon_review_summaries.csv")  # 替换为你的 CSV 文件路径

# 检查数据格式
print("Data preview:")
print(data_df.head())

# Step 2: 加载预训练的 T5 模型
print("Loading T5 model...")
model_name = "t5-small"  # 选择适合的模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Step 3: 针对每条记录生成优点或缺点总结
def summarize_review(summary_text, task_prefix, max_length=50, num_beams=4):
    """
    使用 T5 模型独立生成每条记录的总结。
    Args:
        summary_text: 单条文本，包含商品总结。
        task_prefix: T5 任务前缀，指示模型执行的总结任务。
        max_length: 生成的最长文本长度。
        num_beams: 用于束搜索，提高生成质量。
    
    Returns:
        str: 针对这条记录生成的总结。
    """
    input_text = task_prefix + summary_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # 生成总结
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# Step 4: 针对每条记录逐条生成总结
positive_summaries = []
negative_summaries = []

print("Generating summaries for positive reviews...")
for summary_text in data_df[data_df["label"] == 1]["summary"]:
    # 针对好评生成总结
    positive_summaries.append(summarize_review(
        summary_text,
        task_prefix="summarize: Highlight the main advantages: ",
        max_length=60
    ))

print("Generating summaries for negative reviews...")
for summary_text in data_df[data_df["label"] == 0]["summary"]:
    # 针对差评生成总结
    negative_summaries.append(summarize_review(
        summary_text,
        task_prefix="summarize: Highlight the main issues: ",
        max_length=60
    ))

# Step 5: 聚合总结内容
def aggregate_summaries(summary_list, task_prefix, max_length=80, num_beams=4):
    """
    将所有逐条生成的总结聚合后整理为一个总体总结。
    Args:
        summary_list: 列表，包含所有独立检索的总结内容。
        task_prefix: T5 任务指令前缀。
        max_length: 总体总结的最大长度。
        num_beams: 用于束搜索，提高生成质量。
    Returns:
        str: 最终生成的专业总结。
    """
    # 将所有总结拼接成大段文本输入
    input_text = " ".join(summary_list)
    input_text = task_prefix + input_text

    # 生成最终聚合总结
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 聚合所有好评总结
print("Aggregating positive reviews...")
final_positive_summary = aggregate_summaries(
    positive_summaries,
    task_prefix="summarize: Provide a professional evaluation based on the positive reviews: ",
    max_length=120
)

# 聚合所有差评总结
print("Aggregating negative reviews...")
final_negative_summary = aggregate_summaries(
    negative_summaries,
    task_prefix="summarize: Provide a professional evaluation based on the negative reviews: ",
    max_length=120
)

# Step 6: 输出最终总结
print("\n==== Expert Summary ====\n")

print("Positive Feedback - Highlights:")
print(final_positive_summary)

print("\nNegative Feedback - Issues:")
print(final_negative_summary)

# Step 7: 保存结果到文件
output_file = "final_expert_summary.txt"
with open(output_file, "w") as f:
    f.write("Positive Feedback - Highlights:\n")
    f.write(final_positive_summary + "\n\n")
    f.write("Negative Feedback - Issues:\n")
    f.write(final_negative_summary)

print(f"\nResults saved to {output_file}")