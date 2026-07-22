"""Domain schema for knowledge-graph construction.

These entity and relationship types constrain the LLM's extraction so the graph
stays consistent and queryable. RELATIONSHIP_PATTERNS additionally constrains
which entity *types* each relationship may connect — this is what lets us reject
semantically wrong edges (e.g. a method "EVALUATED_ON" a model) automatically.

This schema was iterated based on real extraction output: 'Model' was added
after base LLMs (Llama, Qwen) were being mis-typed as 'Method'.
"""

ENTITY_TYPES = [
    "Paper",
    "Author",
    "Method",
    "Model",        # base/foundation models (Llama, Qwen, GPT, ...) — distinct from Method
    "Task",
    "Dataset",
    "Metric",
    "Institution",
]

RELATIONSHIP_TYPES = [
    "AUTHORED_BY",
    "PROPOSES",
    "EXTENDS",
    "USES_METHOD",
    "USES_MODEL",
    "IMPLEMENTS",
    "EVALUATED_ON",
    "ADDRESSES",
    "REPORTS_METRIC",
    "CITES",
]

# Allowed (source_type, target_type) pairs for each relationship. A relationship
# whose endpoints don't match one of these patterns is rejected during validation,
# even if its type is in RELATIONSHIP_TYPES. This enforces direction + domain/range.
RELATIONSHIP_PATTERNS = {
    "AUTHORED_BY":    {("Paper", "Author")},
    "PROPOSES":       {("Paper", "Method")},
    "EXTENDS":        {("Method", "Method")},
    "USES_METHOD":    {("Paper", "Method"), ("Method", "Method")},
    "USES_MODEL":     {("Paper", "Model"), ("Method", "Model")},
    "IMPLEMENTS":     {("Method", "Method"), ("Paper", "Method")},
    "EVALUATED_ON":   {("Paper", "Dataset"), ("Method", "Dataset")},
    "ADDRESSES":      {("Paper", "Task"), ("Method", "Task")},
    "REPORTS_METRIC": {("Paper", "Metric"), ("Method", "Metric")},
    "CITES":          {("Paper", "Paper")},
}