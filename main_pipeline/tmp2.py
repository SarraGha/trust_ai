from typing import List

import spacy
from spacy.tokens import Token

if __name__ == "__main__":

    nlp = spacy.load("en_core_web_sm")


    def tag_predicate_roles(sentence: str):
        doc = nlp(sentence)
        tokens = next(doc.sents)

        predicate = None
        for c in tokens.root.children:
            if c.dep_ not in ("nsubj",):
                predicate = c

        assert predicate is not None, "Please, proper error management"

        def split(predicate: Token) -> List[Token]:
            children = list(predicate.children)
            out = []
            if predicate.dep_ == "prep":
                out.append(predicate)
            elif predicate.dep_ == "pobj":
                out.append(list(predicate.subtree))
                return out

            for c in children:
                tokens = split(c)
                out.extend(tokens)

            return out

        out = split(predicate)
        return ""


    # Example usage

    sentence = "The quick brown fox jumps over the lazy dog"
    tagged = tag_predicate_roles(sentence)
    print(tagged)

    doc = nlp(sentence)
    sentence = next(doc.sents)

    sentence = "Additionally, deforestation, industrial activities, and farming methods emit substantial quantities of carbon dioxide, methane, and nitrous oxide, which retain heat in the atmosphere and result in global warming."
    tagged = tag_predicate_roles(sentence)
    print(tagged)

    print()

    sentence = "The greatest factor is the combustion of fossil fuels—such as coal, oil, and natural gas—used for energy and transportation."
    tagged = tag_predicate_roles(sentence)
    print(tagged)

    sentence = "After the storm passed, the volunteers began clearing debris from the roads."
    tagged = tag_predicate_roles(sentence)
    print(tagged)

    sentence = "The machine learning algorithm outperformed traditional models."
    tagged = tag_predicate_roles(sentence)
    print(tagged)

    sentence = "When we study hard, we usually do well."
    tagged = tag_predicate_roles(sentence)
    print(tagged)

    # Visualisation
    # https://demos.explosion.ai/displacy?text=The%20quick%20brown%20fox%20jumps%20over%20the%20lazy%20dog.&model=en_core_web_sm&cpu=1&cph=1
