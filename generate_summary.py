from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

def summarize_reviews(input_file, output_file, model_name="t5-small"):
    """
    This function generates summaries for product reviews stored in a CSV file.

    Args:
        input_file: Path to the CSV file containing cleaned product reviews.
        output_file: Path to save the new CSV file with summaries.
        model_name: Name of the Transformer model, default is "t5-small".

    It reads the input reviews, uses a Transformer model to summarize them,
    and writes the results to a new file.
    """
    
    # Step 1: Load the Transformer model and tokenizer
    print(f"Loading the model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Model is loaded successfully!")

    # Step 2: Read the input file
    print(f"Reading data from {input_file}")
    data_df = pd.read_csv(input_file)
    print(f"Here are some examples from the dataset:\n{data_df.head()}")

    # Step 3: Function to generate summaries for text inputs
    def generate_summary(text, max_length=30):
        """
        Creates a summary for a single review.

        Args:
            text: The input review to summarize.
            max_length: The length limit for the generated summary.

        Returns:
            A string containing the summary.
        """
        # Prepend the summarization task identifier
        task_prefix = "summarize: "
        input_text = task_prefix + text

        # Tokenize and prepare the input for the model
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate the summary using the model
        summary_ids = model.generate(
            inputs.input_ids, 
            max_length=max_length, 
            num_beams=4,  # Beam search for higher quality
            early_stopping=True  # Stop when sufficient text is generated
        )
        
        # Decode the tokens into a readable string
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    # Step 4: Generate summaries for all reviews
    print("Generating summaries for the reviews...")
    summaries = []
    for i, content in enumerate(data_df["text"]):
        print(f"Processing review {i + 1}/{len(data_df)}")
        try:
            summary = generate_summary(content)
            summaries.append(summary)
            print(f"Summary: {summary}")  # Log the summary
        except Exception as e:
            print(f"Error processing review {i + 1}: {e}")
            summaries.append("")  # Keep it empty if an error occurs

    # Step 5: Add the summaries as a new column to the DataFrame
    data_df["summary"] = summaries

    # Step 6: Save the updated reviews to a new file
    data_df.to_csv(output_file, index=False)
    print(f"Summaries saved successfully to {output_file}")


if __name__ == "__main__":
    input_file = "cleaned_amazon_reviews.csv" 
    output_file = "amazon_review_summaries.csv" 
    summarize_reviews(input_file, output_file)  