from typing import List, Dict

from openai import OpenAI


class LLM:

    def __init__(self, client: OpenAI):
        self._client = client

    def query(self, messages: List[Dict[str, str]]) -> str:
        completion = self._client.chat.completions.create(model="gpt-4o", messages=messages)
        return completion.choices[0].message.content.strip()


if __name__ == "__main__":
    prompt = (
        "# Instructions \n\n"
        "Given a text extract, place a factual data in the sentence's objects between brackets [ ]. \n"
        "A fact is a piece of information that is objectively true, measurable, or verifiable. This includes:\n"
        "- Dates (e.g., [July 29th], [1991])\n"
        "- Names of people, places, and organizations (e.g., [Greek], [Pepsi])\n"
        "- Scientific and technical terms (e.g., [greenhouse gases])\n"
        "- Numerical data or direct measurements (e.g., [20%], [one third])\n"
        # "- Objective actions and events that have clear historical, scientific, or legal verification "
        # "(e.g., [discovered], [signed into law])\n"
        # "- Well-established causal relationships, meaning consequences or major contributors that are widely accepted "
        # "and backed by evidence (e.g., Deforestation contributes to [habitat loss])\n"
        "Do not mark vague, general opinions, interpretations, or vague descriptions. "
        "Do not mark the sentence subjects.\n\n"
        "# Example: \n\n"
        "Input: On September 6th, 1609, only five days after the arrival of the first Dutch and English "
        "sailors, John Colman was reportedly killed by attacking Native Americans by an arrow to his neck."
        "\n"
        "Output: On [September 6th, 1609], only [five days] after the arrival of the first [Dutch] and [English] "
        "[sailors], John Colman was [killed by] attacking [Native Americans] [by an arrow] to his [neck]."
        "\n\n"
        "Output a text with brackets around factual data. Do not produce annotations, formatting, or comments, "
        "only raw text."
        "\n\n"
        "Input: {a0_text}"
        "\n"
        "Output: "
    )

    answer = "Human activities are the main driver of climate change, largely due to the rise in greenhouse gas concentrations in the atmosphere. The greatest factor is the combustion of fossil fuels—such as coal, oil, and natural gas—used for energy and transportation. Additionally, deforestation, industrial activities, and farming methods emit substantial quantities of carbon dioxide, methane, and nitrous oxide, which retain heat in the atmosphere and result in global warming."
    prompt = prompt.format(a0_text=answer)

    client = OpenAI()
    llm = LLM(client)

    for _ in range(4):
        response_text = llm.query([{"role": "user", "content": prompt}])
        print(response_text)
        print()
