import os

es_host1 = os.getenv("ELASTIC_HOST1", "172.18.207.05")
es_host2 = os.getenv("ELASTIC_HOST2", "172.18.207.06")
es_host3 = os.getenv("ELASTIC_HOST3", "172.18.207.07")
es_w2v_models_repo = os.getenv("ES_W2V_MODEL_REPO", "ml_models_w2v")
