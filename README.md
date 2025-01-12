# trust_ai
# General Context

# Noise Evaluation Process (main_pipeline)
This section includes scripts for generating an AI-generated dataset, merging AI-generated and human_generated dataset, and launching an evaluation interface for evaluating human-generated and AI-generated responses in order to know if AI performed as good as humand.

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
  - [Generating the Question-Ground Truth File](#generating-the-question-ground-truth-file)
  - [Generating the AI-Generated Dataset](#generating-the-ai-generated-dataset)
  - [Merging the Datasets](#merging-the-datasets)
  - [Launching the Evaluation Interface](#launching-the-evaluation-interface)
- [Input and Output Formats](#input-and-output-formats)
- [Testing Evaluation Directly](#testing-evaluation-directly)

## Requirements
- Python 3.x
- Gradio library

## Usage
### Generating the Question-Ground Truth File
```bash
python extract.py -i <input_file.json> -o <output_file.json>
```
Example:
```bash
python extract.py -i human_dataset.json -o question_groundTruth_dataset.json
```
### enerating the AI-Generated Dataset
```bash
python noise.py -i <input_file.json> -o <output_file.json>
```
Example:
```bash
python noise.py -i question_groundTruth_dataset.json -o noise.json
```
### Merging the Datasets
```bash
python merge_datasets.py <human_dataset.json> <noise.json> <merged_dataset.json>
```
Example:
```bash
python merge_datasets.py human_dataset.json noise.json evaluation_dataset.json
```
### Launching the Evaluation Interface
```bash
python evaluation_interface.py <evaluation_dataset_filename.json>
```
Example:
```bash
python evaluation_interface.py evaluation_dataset.json
```
## Input and Output Formats
Input File Format for extract.py
The input text file should contain questions and their corresponding ground truths and answers in the following format:
```bash
question: What are the main causes of climate change?
ground_truth: Climate change is primarily caused by human activities...
a0: The primary driver of climate change is human activity...
a1: The primary driver of climate change is human activity...
...
```
Output File Format for extract.py
The output JSON file will have the following structure:
```bash
[
    {
        "question": "What are the main causes of climate change?",
        "ground_truth": "Climate change is primarily caused by human activities..."
    },
    {
        "question": "How does photosynthesis work in plants?",
        "ground_truth": "Photosynthesis is the process by which plants convert light energy..."
    }
]
```
Merged Dataset Format
The merged dataset will have the following structure:
```bash
{
   "questions":[
      {
         "id":1,
         "question":"What was the Castlereagh–Canning duel?",
         "ground_truth":"The Castlereagh–Canning duel was a pistol duel...",
         "answers":{
            "A0":{
               "human":"The Castlereagh–Canning duel, fought on September 21, 1809...",
               "ai":"Climate change is predominantly attributed to human actions..."
            },
            ...
         }
      },
      ...
   ]
}
```
## ⚠️ Testing Evaluation Directly
It is possible to test the evaluation interface directly with the dataset available in this repository through this command : 
```bash
python evaluation_interface.py evaluation_dataset.json
```



