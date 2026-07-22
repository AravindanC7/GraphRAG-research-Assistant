"""Evaluation test set — balanced across the corpus's four topical clusters,
with gold sets VERIFIED against the source PDFs.

Clusters:
  uncertainty (TokUR), traffic (LibSignal), magents-unc (MATU), lang-marl (LangMARL)

Each gold item may be a string OR a list of acceptable surface forms (aliases);
the scorer counts the item found if ANY alias appears. This fixes false zeros
from phrasing differences (e.g. "TokUR (TU)" vs "TokUR TU").

Fields:
  question, gold (list | []-control | None-open), item_type, hop, cluster
"""

QUESTIONS = [
    # ---------- Cluster: TokUR (uncertainty) ----------
    {"question": "What datasets is TokUR evaluated on?",
     "gold": ["MATH500", "GSM8K", "DeepScaleR", "HumanEval", "FactScore"],
     "item_type": "Dataset", "hop": "multi", "cluster": "uncertainty"},

    {"question": "Which uncertainty-estimation baselines does TokUR compare against for incorrect-reasoning-path detection?",
     "gold": ["Self-Certainty", "DeepConf", "LLM-Check", "INSIDE", "P(True)",
              ["Predictive Entropy", "PE"], ["Log-Likelihood", "LL"],
              ["Semantic Entropy", "SE"], ["Shifting Attention to Relevance", "SAR"]],
     "item_type": "Method", "hop": "multi", "cluster": "uncertainty"},

    {"question": "What are the ablation variants of TokUR?",
     "gold": [["TokUR TU", "TokUR (TU)", "TU"], ["TokUR AU", "TokUR (AU)", "AU"],
              ["TokUR EU", "TokUR (EU)", "EU"]],
     "item_type": "Method", "hop": "single", "cluster": "uncertainty"},

    {"question": "What is token-level uncertainty estimation?",
     "gold": None, "item_type": None, "hop": "single", "cluster": "uncertainty"},

    # ---------- Cluster: LibSignal (traffic) ----------
    {"question": "Which traffic simulators does LibSignal integrate?",
     "gold": ["SUMO", "CityFlow"],
     "item_type": "Dataset", "hop": "single", "cluster": "traffic"},

    {"question": "Which traffic-control algorithms does LibSignal benchmark?",
     "gold": ["IDQN", "CoLight", "MaxPressure", "FixedTime", "SOTL", "PressLight",
              "MAPG", "IPPO"],
     "item_type": "Method", "hop": "multi", "cluster": "traffic"},

    {"question": "How does CoLight coordinate traffic signals across intersections?",
     "gold": None, "item_type": None, "hop": "single", "cluster": "traffic"},

    # ---------- Cluster: MATU (multi-agent uncertainty) ----------
    {"question": "Which LLM backbones does MATU use for backbone selection?",
     "gold": [["Qwen2.5-7B", "Qwen-2.5-7B"], ["Llama3.1-8B", "Llama-3.1-8B"],
              "Qwen3-4B", "Gemma3-4B"],
     "item_type": "Model", "hop": "multi", "cluster": "magents-unc"},

    {"question": "What baselines does MATU compare against?",
     "gold": ["SAUP", "P(True)", ["EmbeddingSimilarity", "Embedding Similarity"]],
     "item_type": "Method", "hop": "multi", "cluster": "magents-unc"},

    # ---------- Cluster: LangMARL (language MARL) ----------
    {"question": "On which benchmarks is LangMARL evaluated?",
     "gold": ["HotPotQA", "MATH", "HumanEval", ["Overcooked-AI", "Overcooked"], "Pistonball"],
     "item_type": "Dataset", "hop": "multi", "cluster": "lang-marl"},

    {"question": "Which optimization baselines does LangMARL compare against?",
     "gold": ["Reflexion", "TextGrad", ["Auto Prompt Engineer", "AutoPrompt"], "Symbolic"],
     "item_type": "Method", "hop": "multi", "cluster": "lang-marl"},

    # ---------- Cross-paper (traversal's home turf) ----------
    {"question": "Which benchmark datasets are used in more than one paper in this corpus?",
     "gold": ["HumanEval", "MATH", "GSM8K"],
     "item_type": "Dataset", "hop": "multi", "cluster": "cross"},

    {"question": "Which papers or methods evaluate on HumanEval?",
     "gold": ["TokUR", "MATU", "LangMARL"],
     "item_type": None, "hop": "multi", "cluster": "cross"},

    # ---------- Control ----------
    {"question": "What is the capital of France?",
     "gold": [], "item_type": None, "hop": "control", "cluster": "control"},
]