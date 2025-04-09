from typing import Union, Iterator

import spacy
from spacy import Errors
from spacy.symbols import NOUN, PROPN, PRON, ADV
from spacy.tokens import Doc, Span

if __name__ == "__main__":

    nlp = spacy.load("en_core_web_sm")


    def span_boxes(doclike: Union[Doc, Span]) -> Iterator[Span]:
        """
        Detect base noun phrases in the object and adverbs from a dependency parse.
        """
        labels = [
            "oprd",
            "dobj",
            "advmod",
            "npadvmod",
            "pcomp",
            "pobj",
            "dative",
            "appos",
            "attr",
            "ROOT",
        ]
        doc = doclike.doc  # Ensure works on both Doc and Span.
        if not doc.has_annotation("DEP"):
            raise ValueError(Errors.E029)
        np_deps = [doc.vocab.strings.add(label) for label in labels]
        conj = doc.vocab.strings.add("conj")
        prev_end = -1
        for i, word in enumerate(doclike):
            if word.pos not in (NOUN, PROPN, ADV):
                continue
            # Prevent nested chunks from being produced
            if word.left_edge.i <= prev_end:
                continue
            if word.dep in np_deps:
                prev_end = word.i
                yield doc[word.left_edge.i:word.i + 1]
            elif word.dep == conj:
                head = word.head
                while head.dep == conj and head.head.i < head.i:
                    head = head.head
                # If the head is an NP, and we're coordinated to it, we're an NP
                if head.dep in np_deps:
                    prev_end = word.i
                    yield doc[word.left_edge.i:word.i + 1]


    def tag_predicate_roles(sentence: str):
        doc = nlp(sentence)

        idx = []
        for box in span_boxes(doc):
            idx.append((min(b.idx for b in box), max(b.idx + len(b) for b in box)))

        idx.sort(reverse=True)

        assert True, "check no overlap"

        boxed_sentence = sentence
        for start, end in idx:
            boxed_sentence = boxed_sentence[:start] + "[" + boxed_sentence[start:end] + "]" + boxed_sentence[end:]

        return boxed_sentence


    # Example usage

    sentence = "The quick brown fox jumps over the lazy dog"
    tagged = tag_predicate_roles(sentence)
    print(tagged)

    sentence = "Photosynthesis is the mechanism through which plants transform light energy into chemical energy stored as glucose. This process takes place in the chloroplasts of plant cells, where chlorophyll captures sunlight. It involves using carbon dioxide from the atmosphere and water from the soil, resulting in the production of glucose and oxygen as byproducts. The complete chemical equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂."
    tagged = tag_predicate_roles(sentence)
    print(tagged)

    sentence = "Apple is considering at purchasing an English start-up for 1 billion dollars tomorrow at mid day."
    tagged = tag_predicate_roles(sentence)
    print(tagged)

    sentence = "Apple is considering at purchasing an English start-up for 1 billion dollars tomorrow morning."
    tagged = tag_predicate_roles(sentence)
    print(tagged)

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

    sentence = "An English start-up is being acquired by Apple for 1 billion dollars tomorrow morning."
    tagged = tag_predicate_roles(sentence)
    print(tagged)

    sentence = "The Castlereagh–Canning duel was a pistol duel fought on September 21, 1809, between the British Minister of War, Viscount Castlereagh, and Foreign Secretary, George Canning, at Putney Heath. The reasons for the duel were the rivalry between the two politicians and numerous disagreements between them over the conduct of the war against Napoleonic France in 1808 and 1809. Castlereagh wounded Canning in the leg, and the incident led to the collapse of the Portland government and the advancement of Spencer Perceval as the new Prime Minister. Castlereagh and Canning, meanwhile, spent several years on the backbenches, absent from any government responsibility."
    tagged = tag_predicate_roles(sentence)
    print(tagged)

    # Visualisation
    # https://demos.explosion.ai/displacy?text=The%20quick%20brown%20fox%20jumps%20over%20the%20lazy%20dog.&model=en_core_web_sm&cpu=1&cph=1
