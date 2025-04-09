from openai import OpenAI

from models import Report, Tracker
from noise_pipeline import CreateNoiseExamplesStep, LLM

if __name__ == "__main__":
    with open("./report.json", "r") as f:
        report = Report.model_validate_json(f.read())

    sample = report.questions[3]

    client = OpenAI()
    llm = LLM(client)
    step = CreateNoiseExamplesStep(llm)

    step.step(sample, Tracker())
    pass
