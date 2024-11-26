import os
from ragas import evaluate, EvaluationDataset
from ragas.metrics import AnswerCorrectness
import pandas as pd

OPENAI_API_KEY = os.getenv('OPENAI_KEY')

# Test data
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
  },
  {
      'user_input': 'Who invented the telephone?',
      'response': 'The telephone was invented by Antonio Meucci in 1871.',
      'reference': 'The telephone was invented by Alexander Graham Bell in 1876.'
  },

  {
      'user_input': 'Explain Einstein s theory of relativity.',
      'response': 'Einstein developed the theory of relativity, which explains how gravity affects space and time and introduces the concept of E=mc^2.',
      'reference': 'Einstein s theory of relativity includes both special and general relativity, revolutionizing the understanding of space, time, and gravity.'
  }

]

# Answer Correctness() metric
metrics = [AnswerCorrectness()]

dataset = EvaluationDataset.from_dict(data_samples)
results = evaluate(dataset=dataset, metrics=metrics)

df = results.to_pandas()
print("\nEvaluation Results:")
print(df)

