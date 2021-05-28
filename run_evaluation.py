import argparse
import os
import pandas as pd
from gensim.models import KeyedVectors


# Similarity-----------------------------------------------------


def evaluate_similarity(path, model):
    data_set = path.replace(".csv", "")[path.replace(".csv", "").rfind("/") + 1 :]
    data = pd.read_csv(path)
    cos_sims = []
    n = data.shape[0]
    for index in range(data.shape[0]):
        try:
            cos_sim = model.similarity(data.iloc[index]["word1"].lower(), data.iloc[index]["word2"].lower())
        except KeyError:
            print("No word {0} or {1} :(".format(data.iloc[index]["word1"], data.iloc[index]["word2"]))
            cos_sim = 0
            n -= 1

        cos_sims.append(cos_sim)

    data["cos_sim"] = cos_sims
    result = (data_set + " mean_cos_sim", sum(cos_sims) / n)

    # при желаний сохранить результаты в таблицу
    # data.to_csv('./results/{}_cos_sim.csv'.format(data_set), index=False)
    return result


# Analogies-----------------------------------------------------


def get_analogy(example, query, emb_model, top_n=10):
    word_positive = [query.lower(), example[1].lower()]
    word_negative = [example[0].lower()]

    analogy = emb_model.most_similar(positive=word_positive, negative=word_negative, topn=top_n)
    return [el[0] for el in analogy]


def get_analogy_by_row(row, model):
    example = [row["query"], row["answer"]]
    query = row["word1"]
    try:
        pred_answer = get_analogy(example, query, model)
    except KeyError:
        pred_answer = []
        print("word {} is not in vocab:(".format(example[0]))
    return pred_answer


def evaluate_analogies(path, model):
    model.init_sims(replace=True)
    data_set = path.replace(".csv", "")[path.replace(".csv", "").rfind("/") + 1 :]
    data = pd.read_csv(path)
    data["pred"] = data.apply(get_analogy_by_row, args=(model,), axis=1)
    data["is_accurate"] = data.apply(lambda r: 1 if r.word2.lower() in r.pred else 0, axis=1)

    # при желаний сохранить результаты в таблицу
    # data.to_csv('./results/{}.csv'.format(data_set), index=False)
    accuracy = data["is_accurate"].sum() / len(data) * 100
    return "Accuracy_percent {}".format(data_set), accuracy


def get_model_statistics(model_path, list_of_sim_datasets, list_of_anal_datasets):
    print("Model loading...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    statistics = []

    print("Calculating cosine similarity")
    for ds in list_of_sim_datasets:
        statistics.append(evaluate_similarity(ds, model))

    print("Calculating analogies")
    for ds in list_of_anal_datasets:
        statistics.append(evaluate_analogies(ds, model))

    del model

    return statistics


def main(model_path: str, list_of_sim_datasets, list_of_anal_datasets):
    statistics = get_model_statistics(model_path, list_of_sim_datasets, list_of_anal_datasets)

    columns = [model_path]

    cos_sim_values = [test[1] for test in statistics if test[0].endswith("cos_sim")]
    analogy_accuracy_values = [test[1] for test in statistics if test[0].startswith("Accuracy")]

    mean_cos_sim_terms = sum(cos_sim_values) / len(cos_sim_values)
    mean_acc_anal_terms = sum(analogy_accuracy_values) / len(analogy_accuracy_values)

    data_mean_result = pd.DataFrame(
        [mean_cos_sim_terms, mean_acc_anal_terms], columns=columns, index=["mean_cos_sim", "mean_acc_analogies"]
    )
    print('saving mean results to "./results/results.csv"...')
    data_mean_result.to_csv("./results/results.csv")

    # при желании сохранить статистку по всем наборам отдельно
    # index = [test[0] for test in statistics]
    # data_extended = pd.DataFrame([i[1] for i in statistics],
    #                              columns=columns, index=index)
    # data_extended.to_csv('./results/results_extended.csv')

    return statistics


if __name__ == "__main__":

    # PARAMETERS--------------------------------------------------

    # обязательный аргумент - путь к модели
    parser = argparse.ArgumentParser(description="path to model to test embedding qualities")
    parser.add_argument("--model-path", required=True, help="path to word2vec model")

    # можно  добавить путь к новым данным в формате csv с колонками: 'word1', 'word2'
    parser.add_argument("--sim-data", nargs="+", help="new paths to additional data to check cos sim")

    # можно  добавить путь к новым данным в формате csv с колонками: 'query', 'answer', 'word1', 'word2'
    parser.add_argument("--an-data", nargs="+", help="new paths to additional data to check analogies")

    args = parser.parse_args()

    # список наборов данных для оценки семантической близости спомощью cosine similarity
    LIST_OF_SIM_DATASETS = [
        "./data/similarity/Economics.csv",
        "./data/similarity/Politics.csv",
        "./data/similarity/It.csv",
        "./data/similarity/Science.csv",
        "./data/similarity/Med.csv",
    ]

    # список наборов данных для оценки поиска аналогий спомощью cosine similarity
    LIST_OF_ANALOG_DATASETS = [
        "./data/analogies/Economics_anal.csv",
        "./data/analogies/Politics_anal.csv",
        "./data/analogies/It_anal.csv",
        "./data/analogies/Science_anal.csv",
        "./data/analogies/Med_anal.csv",
    ]

    MODEL_PATH = args.model_path

    if not os.path.exists("results"):
        os.makedirs("results")

    if args.sim_data:
        LIST_OF_SIM_DATASETS.extend(args.sim_data)

    if args.an_data:
        LIST_OF_ANALOG_DATASETS.extend(args.an_data)

    # EVALUATION-------------------------------------------------

    stat = main(MODEL_PATH, LIST_OF_SIM_DATASETS, LIST_OF_ANALOG_DATASETS)
    print("model statistics:\n")
    for i in stat:
        print("{0}: {1}".format(*i))
