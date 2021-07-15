import collections
import itertools
import logging
import statistics
from gensim.models import KeyedVectors

# from data.analogies.en import ANALOGIES
# from data.similarity.en import SIM

logger = logging.getLogger(__name__)
missing_terms = set()


def evaluate_similarity(model, similarities_collection):
    global missing_terms
    sim_overall = []
    sim_groups = collections.defaultdict(list)

    missing_overall = 0
    missing_groups = {}

    for area, similarities in similarities_collection.items():
        missing = 0
        for pair in similarities:
            cos_sim = 0
            cartesian_product = list(itertools.product(pair["word1"], pair["word2"]))
            attempts = 0
            for word1, word2 in cartesian_product:
                try:
                    cos_sim = max(cos_sim, model.similarity(word1, word2))
                except KeyError:
                    attempts += 1

            if attempts >= len(cartesian_product):
                for word in set(pair["word1"]).union(pair["word2"]):
                    if word not in model:
                        missing_terms.add(word)
                missing += 1
                missing_overall += 1
            sim_overall.append(cos_sim)
            sim_groups[area].append(cos_sim)
            missing_groups[area] = 1 - missing / len(similarities)
    result = {
        "similarity_overall": statistics.mean(sim_overall),
        "completeness_overall": 1 - missing_overall / sum(map(len, similarities_collection.values())),
        "similarity_by_groups": {k: statistics.mean(v) for k, v in sim_groups.items()},
        "completeness_by_groups": missing_groups,
    }

    return result


def get_analogy_by_row(analogy, model):
    global missing_terms
    predicted = False

    for query, answer, word1, word2 in list(
        itertools.product(analogy["query"], analogy["answer"], analogy["word1"], analogy["word2"])
    ):
        try:
            predicted_answers = model.most_similar(positive=[word1, answer], negative=[query], topn=20)
            if word2 in set(i[0] for i in predicted_answers):
                return True
        except KeyError:
            pass

    if not predicted:
        for word in set(analogy["query"] + analogy["answer"] + analogy["word1"] + analogy["word2"]):
            if word not in model:
                missing_terms.add(word)
    return predicted


def evaluate_analogies(model, analogies_collections):

    accurate_predictions = 0
    accurate_predictions_by_group = collections.defaultdict(int)
    for area, analogies in analogies_collections.items():
        for analogy in analogies:
            predicted = get_analogy_by_row(analogy, model)
            if predicted:
                accurate_predictions += 1
                accurate_predictions_by_group[area] += 1

    result = {
        "analogies_accuracy": accurate_predictions / sum(map(len, analogies_collections.values())),
        "analogies_by_groups": {k: v / len(analogies_collections[k]) for k, v in accurate_predictions_by_group.items()},
    }

    return result


def expand_word_set(words):
    return list(set(words + list(map(lambda x: x.lower(), words))))


def enhance_collection(collection, fields):
    enhanced = {}
    for area, dataset in collection.items():
        enhanced[area] = []
        for item in dataset:
            for field in fields:
                item[field] = expand_word_set(item[field])
            enhanced[area].append(item)
    return enhanced


def get_model_statistics(model_path, lang):

    if lang == "en":
        from data.analogies.en import ANALOGIES
        from data.similarity.en import SIM
    elif lang == "ru":
        from data.analogies.ru import ANALOGIES
        from data.similarity.ru import SIM
    else:
        logger.info("This language is not supported yet")
        return

    logger.info("Enhancing datasets")
    similarities_enhanced = enhance_collection(SIM, fields=["word1", "word2"])
    analogies_enhanced = enhance_collection(ANALOGIES, fields=["word1", "word2", "answer", "query"])

    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    logger.info("Calculating cosine similarity")
    similarity_stats = evaluate_similarity(model, similarities_collection=similarities_enhanced)

    logger.info("Calculating analogies")
    analogies_stats = evaluate_analogies(model, analogies_collections=analogies_enhanced)

    del model

    return {"similarity": similarity_stats, "analogies": analogies_stats}
