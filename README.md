## Synthetic Data Generation for Stage 1 (After Download datasets by BEIR, dataset=hotpotqa, fiqa, scifact, nfcorpus)
cd datasets
python parse_synthetic.py {dataset}

## Stage 1
python train_stage1.py e5 {dataset}

## Retrieval
python retrieval.py e5-sft {dataset}

## Pair-wise comparison
cd stage2
python parse_comparison.py {dataset}

## Stage 2
python train_stage2.py e5-sft {dataset}

## Evaluation
python evaluate.py e5-final {dataset}
