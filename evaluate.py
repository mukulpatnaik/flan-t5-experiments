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

def evaluate_mmlu(dataset, model, tokenizer, name):
    print("Evaluating model: ", name)
    answers = []
    score = 0
    errors = []
    results = []
    for i in range(len(dataset['question'])):

        # The choices should be in the format of "A: choice1, B: choice2, C: choice3, D: choice4" and be seperated from the question by a tab
        prompt = dataset['question'][i] + "\n" + "A: " + dataset['choices'][i][0] + ", B: " + dataset['choices'][i][1] + ", C: " + dataset['choices'][i][2] + ", D: " + dataset['choices'][i][3]
        # print(prompt)

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10)
        a = tokenizer.decode(outputs[0])
    
        x = a.replace('<pad>', '').replace('</s>', '')
        z = x.strip()

        answers.append(z)
        keys = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        # append the question, answer, and correct answer to the results list as a dictionary
        results.append({'question': dataset['question'][i], 'answer': z, 'correct_answer': list(keys.keys())[list(keys.values()).index(dataset['answer'][i])], 'choices': dataset['choices'][i]})
        # print(keys[str(z)])
        if keys[str(z)] == dataset['answer'][i]:
            score += 1
        elif str(z) not in keys.keys():
            print("Error: ", z)
        else:
            errors.append({'question': dataset['question'][i], 'answer': z, 'correct_answer': list(keys.keys())[list(keys.values()).index(dataset['answer'][i])], 'choices': dataset['choices'][i]})  
    
    print("Score: ", score/len(answers))
    return results, errors

def run_mmlu(dataset, model_size="base", n=100):
    # model_size = str(input("Enter model size: "))
    model, tokenizer = get_model_and_tokenizer(model_size)
    df = pd.DataFrame(dataset['auxiliary_train'])
    g = df.sample(n=n, random_state=42).reset_index(drop=True)
    results, errors = evaluate_mmlu(g, model, tokenizer, model_size)
    return results, errors

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate Flan-T5 model on SAT questions')
    parser.add_argument('model_size', choices=['small', 'base', 'large', 'xl', 'xxl', 'eightbitmodel'], help='Size of T5 model')
    parser.add_argument('-b', '--benchmark', type=str, default='sat', help='the benchmark to use')
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model_size)

    if (args.benchmark == 'mmlu'):

        from datasets import load_dataset
        dataset = load_dataset("hendrycks_test", 'global_facts')
        data = pd.DataFrame(dataset['test'])
        results, errors = run_mmlu(dataset, 'base', 100)
    
    elif (args.benchmark == 'sat'):

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
    results_df.to_csv(f'results/{args.model_size}-{args.benchmark}-results.csv', index=False)

if __name__ == '__main__':
    main()