# Syntriever
Syntriever: How to Train Your Retriever with Synthetic Data from LLMs

```
## Synthetic Data Generation for Stage 1 (After Download datasets by BEIR, dataset=msmarco, hotpotqa, fiqa, scifact, nfcorpus)
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
```


## Citation
If you use any part of this code and pretrained weights for your own purpose, please cite our [paper]().
```
@InProceedings{
  title = 	 {Syntriever: How to Train Your Retriever with Synthetic Data from LLMs},
  author =       {Minsang Kim, Seungjun Baek},
  booktitle = 	 {Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), Findings},
  year = 	 {2025},
  series = 	 {Proceedings of Findings of NAACL},
  month = 	 {30, April -- 2 May},
  publisher =    {Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), Findings}},
  pdf = 	 {},
  abstract = 	 {LLMs have boosted progress in many AI applications. Recently, there were attempts to distill the vast knowledge of LLMs into information retrieval systems. Those distillation methods mostly use output probabilities of LLMs which are unavailable in the latest black-box LLMs. We propose Syntriever, a training framework for retrievers using synthetic data from black-box LLMs. Syntriever consists of two stages. Firstly in the distillation stage, we synthesize relevant and plausibly irrelevant passages and augmented queries using chain-of-thoughts for the given queries. LLM is asked to self-verify the synthetic data for possible hallucinations, after which retrievers are trained with a loss designed to cluster the embeddings of relevant passages. Secondly in the alignment stage, we align the retriever with the preferences of LLMs. We propose a preference modeling called partial Plackett-Luce ranking to learn LLM preferences with regularization which prevents the model from deviating excessively from that trained in the distillation stage. Experiments show that Syntriever achieves state-of-the-art performances on benchmark datasets from various domains in nDCG@K$.}
  }
```
