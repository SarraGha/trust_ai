import os
from llama_index.llms.ollama import Ollama
from ragas.llms import LlamaIndexLLMWrapper
from ragas.metrics import AnswerCorrectness
from ragas import evaluate, EvaluationDataset
import pandas as pd

# Set up the Ollama client with the specified host
os.environ['BASE_URL'] = "http://atlas1api.eurecom.fr:8019"

# Ollama model
llm = Ollama(model="llama3.1:70b", base_url=os.environ['BASE_URL'])

evaluator_llm = LlamaIndexLLMWrapper(llm)

#metrics to be used for evaluation
metrics = [
  AnswerCorrectness(llm=evaluator_llm),
]

# Create the evaluation dataset with required columns
data_samples = [
  {
      'user_input': 'When was the first super bowl?',
      'response': 'The first superbowl was held on Jan 15, 1967',
      'reference': 'The first superbowl was held on January 15, 1967'
  },
  {
      'user_input': 'Who won the most super bowls?',
      'response': 'The most super bowls have been won by The New England Patriots',
      'reference': 'The New England Patriots have won the Super Bowl a record six times'
  }
]
dataset = EvaluationDataset.from_dict(data_samples)

# Evaluate the dataset
results = evaluate(dataset=dataset, metrics=metrics)

# Convert results to a pandas DataFrame and display
df = results.to_pandas()
print(df.head())

# Test with additional input
additional_data = [
  {
      'user_input': 'Who invented the telephone?',
      'response': 'The telephone was invented by Antonio Meucci in 1871.',
      'reference': 'The telephone was invented by Alexander Graham Bell in 1876.'
  }
]
additional_dataset = EvaluationDataset.from_dict(additional_data)

# Evaluate the additional dataset
additional_results = evaluate(dataset=additional_dataset, metrics=metrics)

# Convert additional results to a pandas DataFrame and display
additional_df = additional_results.to_pandas()
print(additional_df.head())