import argparse
import json
import logging
import pathlib
import pprint

import elasticsearch

import config
from src.eval import get_model_statistics, missing_terms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger("elasticsearch").setLevel(logging.WARN)
logger = logging.getLogger("Расчёт качества векторной модели")

es = elasticsearch.Elasticsearch(
    hosts=[
        {"host": config.es_host1, "port": 9200},
        {"host": config.es_host2, "port": 9200},
        {"host": config.es_host3, "port": 9200},
    ]
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_meta_path", required=True, help="Обязательный аргумент - путь к метаданным модели")
    parser.add_argument(
        "--update_model_stats", required=True, help="Загружать ли статистику модели в репозиторий", type=bool
    )
    parser.add_argument("--language", required=True, help="Язык модели")
    args = parser.parse_args()

    meta = json.loads(pathlib.Path(args.model_meta_path).read_text())

    model_path = str(pathlib.Path(*pathlib.Path(args.model_meta_path).parts[:-1]).joinpath(meta["model_file"]))
    collection_name = meta["model_name"].replace("-", "_")

    statistics = get_model_statistics(model_path, args.language)
    print("model statistics:\n")
    pprint.pprint(statistics)
    print("Missing terms:\n")
    print("\n".join(list(missing_terms)))
    if args.update_model_stats:
        es.update(
            index=config.es_w2v_models_repo,
            id=collection_name,
            body={"doc": {"performance": statistics}},
            request_timeout=3000,
        )
