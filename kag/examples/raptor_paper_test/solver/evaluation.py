# !usr/bin/env python
# -*- coding:utf-8 _*-
import logging
import os
import json

from kag.common.env import init_kag_config
from kag.solver.logic.solver_pipeline import SolverPipeline
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm

logger = logging.getLogger(__name__)

class KagDemo:

    def __init__(self):
        pass

    def qa(self, query):
        resp = SolverPipeline()
        answer, traceLog = resp.run(query)
        rerank_docs = []

        for history_log in traceLog:
            rerank_docs += history_log.get("rerank_docs")

        logger.info(f"\n\nso the answer for '{query}' is: {answer}\n\n")
        return answer, traceLog, rerank_docs
        
def process_question(args):
    """处理单个问题"""
    question, evalObj = args
    answer, trace_log, context = evalObj.qa(question)
    return answer, trace_log, context

if __name__ == "__main__":
    evalObj = KagDemo()

    with open(r"solver\data\raptor_paper_qa.json", "r", encoding="utf-8") as f:
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
    train_data["contexts"] = context

    # Save results
    with open(r"solver\data\raptor_paper_qa_rag.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
