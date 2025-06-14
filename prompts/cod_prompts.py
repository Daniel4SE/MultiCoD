"""Additional Chain of Draft prompts and templates"""

# Fallback CoD prompt if dynamic generation fails
FALLBACK_COD_PROMPT = """
Solve this coding task using Chain of Draft methodology.

TASK:
{task_description}

CODE TO IMPLEMENT:
{task_signature}

RULES:
- Each step must be ≤ 5 words
- Focus on code progression only
- 5-10 steps total
- No explanations

Format:
DRAFT:
step1: [max 5 words]
step2: [max 5 words]
...

SOLUTION:
```python
{task_signature}
    # Implementation