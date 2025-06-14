"""Prompts for strategy and code generation"""

STRATEGY_GENERATION_PROMPT = """
You are an expert at solving coding problems. Given a coding task, generate 5 diverse Chain of Draft (CoD) strategies.

TASK DESCRIPTION:
{task_description}

FUNCTION SIGNATURE:
{task_signature}

Generate 5 different strategies for solving this task. Each strategy should have:
- A unique name
- A specific focus area
- Clear instructions for implementation
- Key priorities

Return your response as a JSON object with this format:
{{
    "strategies": [
        {{
            "name": "Strategy Name",
            "focus": "What this strategy focuses on",
            "instruction": "How to approach the problem with this strategy",
            "key_priorities": ["priority1", "priority2", "priority3"]
        }},
        ... (4 more strategies)
    ]
}}

Make each strategy distinct and suitable for the Chain of Draft methodology where each step is ≤ 5 words.
"""

COD_GENERATION_PROMPT = """
Solve this coding task using the Chain of Draft methodology with the specified strategy.

TASK DESCRIPTION:
{task_description}

FUNCTION SIGNATURE:
{task_signature}

STRATEGY: {strategy_name}
FOCUS: {strategy_focus}
INSTRUCTION: {strategy_instruction}

RULES:
1. Create a DRAFT with 5-10 steps
2. Each step must be ≤ 5 words
3. Steps should follow the strategy's focus and instruction
4. After the draft, provide the complete Python solution

Format your response as:

DRAFT:
step1: [max 5 words describing first action]
step2: [max 5 words describing second action]
...

SOLUTION:
```python
{task_signature}
    # Your implementation here
```

Remember: Focus on {strategy_focus} and follow the instruction: {strategy_instruction}
"""