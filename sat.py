import argparse
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration

def get_model_and_tokenizer(model_size):
    if model_size in ["small", "large", "base", "xl", "xxl"]:
        model = T5ForConditionalGeneration.from_pretrained(f"google/flan-t5-{model_size}")
        tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{model_size}")
    elif model_size == "eightbitmodel":
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", load_in_8bit=True)
        model.to("mps")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl").input_ids.to("mps")
    else:
        raise ValueError(f"Invalid model : {model_size}")

    return model, tokenizer

def evaluate(model, tokenizer, df, name):
    print("Evaluating model: ", name)
    answers = []
    score = 0
    results = []
    for i in df['text']:
        inputs = tokenizer(i, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10)
        answers.append(tokenizer.decode(outputs[0]))
    
    for i in range(len(answers)):
        x = answers[i].replace('<pad>', '').replace('</s>', '')
        a = x.strip()
        # append the question, answer, and correct answer to the results list as a dictionary
        results.append({'question': df['text'][i], 'answer': a, 'correct_answer': df['answer'][i]})
        if a == df['answer'][i]:
            score += 1
    
    print("Score: ", score/len(answers))
    return results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate Flan-T5 model on SAT questions')
    parser.add_argument('model_size', choices=['small', 'base', 'large', 'xl', 'xxl', 'eightbitmodel'], help='Size of T5 model')
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model_size)

    # Read train data from parquet file in data/train-00000-of-00001-be16864a4346f8b0.parquet
    train = pd.read_parquet('data/train-00000-of-00001-be16864a4346f8b0.parquet')
    # Read test data from parquet file in data/test-00000-of-00001-8026e2bb5cef708b.parquet
    test = pd.read_parquet('data/test-00000-of-00001-8026e2bb5cef708b.parquet')
    # Read validation data from parquet file in data/validation-00000-of-00001-6242383510343be0.parquet
    validation = pd.read_parquet('data/validation-00000-of-00001-6242383510343be0.parquet')

    # Combine train, test, and validation data
    data = pd.concat([train, test, validation])

    # Randomly sample 100 rows from data and reset index
    data = data.sample(n=1, random_state=42).reset_index(drop=True)

    # Evaluate
    results = evaluate(model, tokenizer, data, args.model_size)

    # Save results to CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results/{args.model_size}-SAT-results.csv', index=False)

if __name__ == '__main__':
    main()