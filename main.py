import pathlib
from pyexpat.errors import messages

import ollama
from nltk.app.wordnet_app import explanation

if __name__ == "__main__":
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")

    response = client.chat(model="llama3.1", messages=[{"role":"user", "content":"hello"}])

    print(response["message"]["content"])

    prompt = pathlib.Path('file').read_text()
    content = prompt.format(respoonse='')

    responsePrometheus = client.chat('',messages=[{"role":"system", "content":"You are a fair evaluator language model."},
                             {"role":"user","content":content}])
    feedbqck = responsePrometheus["message"]["content"]
    explanation, score = feedbqck.split("[RESULT]")
    scoreInt= int(score.strip())