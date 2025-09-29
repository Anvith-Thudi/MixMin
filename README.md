# MixMin
Code Base for ["MixMin: Finding Data Mixtures via Convex Minimization"](https://arxiv.org/abs/2502.10510) published at ICML 2025.

The code for the LLM experiments is under "LLM_DataMixing", and the workflow is:

1) train proxy models (run train.py with desired arguments)
2) eval proxy models on target (run llm_eval.py)
3) collect logprobs from eval (run logprobs.py)
4) run mixmin (run mixmin_gen.py)
5) train final model (run train.py with mixture weights pointing to the mixmin weight)

The code for the chemistry experiments is under "PCBA", and the workflow is (analogous to above):

1) train proxy models(run pcba_100k_models.py)
2) run mixmin (run pcba_100k_mixmin.py)
3) train final model (run pcba_100k_mixmin_train.py)

If adapting to a new dataset, I recommend adapting the pcba codebase (which has cleaner functions) unless dealing specifically with LLMs.

*NOTE*: I used the environment in mixmin_requirements.txt for everything except training and evaluating LLMs, where I used the llm_requirements.txt in the LLM_DataMixing folder. Both of the requirements.txt have some unecessary packages, and I recommend looking at the imports for the files to have a more streamlined environment.




