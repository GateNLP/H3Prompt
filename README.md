# **Document Classification Script**

This script `demo.py` classifies a given document into a category, identifies relevant narratives, and determines sub-narratives using a fine-tuned language model.

## **Usage**
```bash
python demo.py <document_path> <model_path>
```

- `<document_path>`: Path to the text file containing the document.
- `<model_path>`: Path to the fine-tuned model directory.

## **Output Format**
The script prints the results in the following format:
```
Filename: <document_path>
Category: <Category>
Narratives (Format - Category: Main Narrative: Sub-Narrative):
 - <Category>: <Main Narrative>: <Sub-Narrative>
 - ...
-------------------------
```
For example:
```
Filename: sample.txt
Category: Ukraine-Russia War
Narratives (Format - Category: Main Narrative: Sub-Narrative):
 - CC: Criticism of climate movement: Climate movement is corrupt
 - CC: Criticism of climate movement: Climate movement is alarmist
 - CC: Questioning the measurements and science: Scientific community is unreliable
-------------------------
```

## **Classification Process**
1. **Category Classification:** Determines if the document belongs to:
   - "Ukraine-Russia War" (URW)
   - "Climate Change" (CC)
   - "Other" (if no relevant category applies)

2. **Main Narrative Identification:** Selects the most relevant narrative(s) based on predefined categories.

3. **Sub-Narrative Identification:** Further classifies into specific sub-narratives.

## **Dependencies**
- Python 3.x
- `unsloth` for model inference
- JSON files in `Dataset/` for narrative mappings
