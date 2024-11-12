import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from kag.common.benchmarks.evaluate import Evaluate
from kag.common.env import init_kag_config
from kag.solver.logic.solver_pipeline import SolverPipeline

from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EvaForFrames:

    """
    init for kag client
    """
    def __init__(self, configFilePath):
        self.configFilePath = configFilePath
        init_kag_config(self.configFilePath)

    """
        qa from knowledge base, 
    """
    def qa(self, query):
        # CA
        resp = SolverPipeline()
        answer, traceLog = resp.run(query)
        rerank_docs = []

        for history_log in traceLog:
            rerank_docs += history_log.get("rerank_docs")

        logger.info(f"\n\nso the answer for '{query}' is: {answer}\n\n")
        return answer, traceLog, rerank_docs

    """
        parallel qa from knowledge base
        and getBenchmarks(em, f1, answer_similarity)
    """
    def parallelQaAndEvaluate(
        self, qaFilePath, resFilePath, threadNum=1, upperLimit=10
    ):
        def process_sample(data):
            try:
                sample_idx, sample = data
                sample_id = sample["_id"]
                question = sample["question"]
                gold = sample["answer"]
                prediction, traceLog = self.qa(question)

                evaObj = Evaluate()
                metrics = evaObj.getBenchMark([prediction], [gold])
                return sample_idx, sample_id, prediction, metrics, traceLog
            except Exception as e:
                import traceback

                logger.warning(
                    f"process sample failed with error:{traceback.print_exc()}\nfor: {data}"
                )
                return None

        qaList = json.load(open(qaFilePath, "r"))
        total_metrics = {
            "em": 0.0,
            "f1": 0.0,
            "answer_similarity": 0.0,
            "processNum": 0,
        }
        with ThreadPoolExecutor(max_workers=threadNum) as executor:
            futures = [
                executor.submit(process_sample, (sample_idx, sample))
                for sample_idx, sample in enumerate(qaList[:upperLimit])
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="parallelQaAndEvaluate completing: ",
            ):
                result = future.result()
                if result is not None:
                    sample_idx, sample_id, prediction, metrics, traceLog = result
                    sample = qaList[sample_idx]

                    sample["prediction"] = prediction
                    sample["traceLog"] = traceLog
                    sample["em"] = str(metrics["em"])
                    sample["f1"] = str(metrics["f1"])

                    total_metrics["em"] += metrics["em"]
                    total_metrics["f1"] += metrics["f1"]
                    total_metrics["answer_similarity"] += metrics["answer_similarity"]
                    total_metrics["processNum"] += 1

                    if sample_idx % 50 == 0:
                        with open(resFilePath, "w") as f:
                            json.dump(qaList, f)

        with open(resFilePath, "w") as f:
            json.dump(qaList, f)

        res_metrics = {}
        for item_key, item_value in total_metrics.items():
            if item_key != "processNum":
                res_metrics[item_key] = item_value / total_metrics["processNum"]
            else:
                res_metrics[item_key] = total_metrics["processNum"]
        return res_metrics

def process_question(args):
    """处理单个问题"""
    question, evalObj = args
    answer, trace_log, context = evalObj.qa(question)
    return answer, trace_log, context

if __name__ == "__main__":
    configFilePath = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../kag_config.cfg"
    )
    evalObj = EvaForFrames(configFilePath=configFilePath)

    with open(r"solver\data\frames_data_qa.json", "r") as f:
        train_data = json.load(f)

    num_processes = multiprocessing.cpu_count() - 1
    args_list = [(q, evalObj) for q in train_data["question"]]
    
    # 使用进程池并行处理，带进度条
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_question, args_list),
            total=len(args_list),
            desc="Processing questions"
        ))
    
    answers, trace_log, context = zip(*results)
    train_data["answer"] = answers
    train_data["context"] = context

    # Save results
    with open(r"solver\data\frames_data_qa_rag.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
