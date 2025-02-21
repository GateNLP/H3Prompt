import json
import argparse
from unsloth import FastLanguageModel

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)

def load_model(model_path, max_seq_length=6000, dtype=None, load_in_4bit=False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def unsloth_llm_call(prompt, model, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=500, use_cache=True, do_sample=False)
    return tokenizer.batch_decode(outputs)[0].split("<|start_header_id|>assistant<|end_header_id|>")[-1][:-10].strip()

def classify_category(document_text, model, tokenizer):
    prompt = f"""
    Given the following document text, classify it into one of the two categories: "Ukraine-Russia War" or "Climate Change". 

    Document Text: 
    {document_text}

    Determine the category that closely or partially fits the document. If neither category applies, return "Other". Return only the output, without any additional explanations or text.
    """
    return unsloth_llm_call(prompt, model, tokenizer).replace('"','').replace('.','').strip()

def classify_narratives(document_text, category, narratives, main_narratives_with_explanations, model, tokenizer):
    narratives_list_with_explanations = "\n".join(
        f'- {narrative}: {main_narratives_with_explanations[narrative]}'
        for narrative in narratives[category]
    )
    prompt = f"""
    The document text given below is related to "{category}". 
    Please classify the document text into the most relevant narratives. Below is a list of narratives along with their explanations:

    {narratives_list_with_explanations}

    Document Text: 
    {document_text}
    
    Return the most relevant narratives as a hash-separated string (eg. Narrative1#Narrative2..). If no specific narrative can be assigned, just return "Other" and nothing else. Return only the output, without any additional explanations or text.
    """
    return unsloth_llm_call(prompt, model, tokenizer)

def classify_sub_narrative(document_text, category, main_narrative, narratives, sub_narratives_with_explanations, model, tokenizer):
    if main_narrative == "Hidden plots by secret schemes of powerful groups":
        return "Other"
    sub_narratives_list_with_explanations = "\n".join(
        f'- {sub_narrative}: {sub_narratives_with_explanations[sub_narrative]}'
        for sub_narrative in narratives[category][main_narrative]
    )
    prompt = f"""
    The document text given below is related to "{category}" and its main narrative is: "{main_narrative}".
    Please classify the document text into the most relevant sub-narratives. Below is a list of sub-narratives along with their explanations:

    {sub_narratives_list_with_explanations}

    Document Text:
    {document_text}

    Return the most relevant sub-narratives as a hash-separated string (e.g., Sub-narrative1#Sub-narrative2..). If no specific sub-narrative can be assigned, just return "Other" and nothing else. Return only the output, without any additional explanations or text.
    """
    return unsloth_llm_call(prompt, model, tokenizer)

def classify_document(document_text, model, tokenizer, narratives, main_narratives_with_explanations, sub_narratives_with_explanations):
    final_label = []
    category = classify_category(document_text, model, tokenizer)
    if category == "Other":
        final_label.append("Other : Other")
    else:
        if category in narratives:
            main_narratives = classify_narratives(document_text, category, narratives, main_narratives_with_explanations, model, tokenizer)
            if main_narratives == "Other":
                final_label.append("Other : Other")
            else:
                main_narratives_list = [it.strip("-").strip() for it in main_narratives.split("#")]
                all_sub_narratives = {}
                for main_narrative in main_narratives_list:
                    if main_narrative in narratives[category]:
                        sub_narratives = classify_sub_narrative(document_text, category, main_narrative, narratives, sub_narratives_with_explanations, model, tokenizer)
                        sub_narratives_list = [sub.strip("-").strip() for sub in sub_narratives.split("#")]
                        all_sub_narratives[main_narrative] = sub_narratives_list
                for main, subs in all_sub_narratives.items():
                    for sub in subs:
                        if sub in sub_narratives_with_explanations or sub == "Other":
                            final_label.append(f"{main} : {sub}")
    return [category, final_label] if final_label else ["Other", ["Other : Other"]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=str, help="Path to the document file")
    parser.add_argument("model", type=str, help="Path to the fine-tuned model")
    args = parser.parse_args()
    
    # Load resources
    narratives = load_json('Dataset/labels.json')
    main_narratives_with_explanations = load_json('Dataset/main_narratives_with_explanations.json')
    sub_narratives_with_explanations = load_json('Dataset/sub_narratives_with_explanations.json')
    model, tokenizer = load_model(args.model)
    
    with open(args.document, 'r', encoding='utf-8') as file:
        document_text = file.read()
    
    category, predicted = classify_document(document_text, model, tokenizer, narratives, main_narratives_with_explanations, sub_narratives_with_explanations)
    cat_mapping = {"Ukraine-Russia War":"URW", "Climate Change":"CC", "Other":"Other"}
    if "climate" in category.lower():
        category = "Climate Change"
    if "ukraine" in category.lower() or "russia" in category.lower():
        category = "Ukraine-Russia War"
    cat_code = cat_mapping[category]
    if len(predicted) == 1 and predicted[0] == "Other : Other":
        narr_final_joined = "Other"
        subnarr_final_joined = "Other"
    else:
        narr_final = []
        subnarr_final = []
        for i in range(len(predicted)):
            narr, subnarr = predicted[i].split(" : ")
            narr_final.append(cat_code + ": " + narr)
            subnarr_final.append(cat_code + ": " + narr + ": " + subnarr)
        narr_final_joined = ";".join(narr_final)
        subnarr_final_joined = ";".join(subnarr_final)

    print("Filename: ", args.document)
    print("Category: ", category)
    print("Narrative (Format -> Category: Main Narrative: Sub-Narrative): ")
    line = f"{args.document}\t{narr_final_joined}\t{subnarr_final_joined}" #output format needed for semeval task
    extract_narratives = line.split("\t")[-1].strip().split(";")
    for it in extract_narratives:
        print(it)

if __name__ == "__main__":
    main()
