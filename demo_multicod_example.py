#!/usr/bin/env python3
"""
Demonstration of Multi-CoD Process for Flask Email Task
"""

import anthropic
import subprocess
import tempfile
import os

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initial task prompt
INITIAL_PROMPT = """Creates a Flask application configured to send emails using Flask-Mail. 
It sets up the necessary SMTP configuration dynamically based on provided parameters 
and defines a route to send a test email.
The function should output with:
    Flask: A Flask application instance configured for sending emails.
You should write self-contained code starting with:
```
from flask import Flask
from flask_mail import Mail, Message
def task_func(smtp_server, smtp_port, smtp_user, smtp_password, template_folder):"""

print("="*60)
print("MULTI-COD DEMONSTRATION: Flask Email Task")
print("="*60)

# Step 1: Generate 5 Chain-of-Draft strategies
print("\n1. GENERATING 5 CHAIN-OF-DRAFT STRATEGIES:")
print("-"*40)

strategies_prompt = f"""
Given this coding task:
{INITIAL_PROMPT}

Generate 5 different Chain of Draft (CoD) strategies for solving this task. Each strategy should have a unique approach.

Return as JSON:
{{
    "strategies": [
        {{
            "name": "Strategy Name",
            "focus": "What this strategy focuses on",
            "draft_steps": ["step1", "step2", "step3", "step4", "step5"]
        }}
    ]
}}

Each draft step must be ≤ 5 words. Generate exactly 5 strategies.
"""

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2000,
    temperature=0.7,
    messages=[{"role": "user", "content": strategies_prompt}]
)

import json
strategies_text = response.content[0].text
json_start = strategies_text.find('{')
json_end = strategies_text.rfind('}') + 1
strategies = json.loads(strategies_text[json_start:json_end])['strategies']

for i, strategy in enumerate(strategies):
    print(f"\nStrategy {i+1}: {strategy['name']}")
    print(f"Focus: {strategy['focus']}")
    print("Draft steps:")
    for j, step in enumerate(strategy['draft_steps']):
        print(f"  step{j+1}: {step}")

# Step 2: Generate 5 solutions using the CoD strategies
print("\n\n2. GENERATING 5 SOLUTIONS:")
print("-"*40)

solutions = []
for i, strategy in enumerate(strategies):
    cod_prompt = f"""
Solve this coding task using Chain of Draft methodology:

{INITIAL_PROMPT}

Use this specific strategy: {strategy['name']}
Focus: {strategy['focus']}

Your draft steps:
{chr(10).join(f"step{j+1}: {step}" for j, step in enumerate(strategy['draft_steps']))}

Now implement the complete solution following these steps.

Return only the code without any explanation.
"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        temperature=0.5 + i*0.1,  # Varying temperature
        messages=[{"role": "user", "content": cod_prompt}]
    )
    
    code = response.content[0].text
    # Extract code between triple backticks if present
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    
    solutions.append(code.strip())
    print(f"\nSolution {i+1} generated using {strategy['name']}")

# Step 3: Evaluate each solution
print("\n\n3. EVALUATING SOLUTIONS:")
print("-"*40)

# Test code for Flask email task
TEST_CODE = """
# Test the function
try:
    app = task_func('smtp.gmail.com', 587, 'user@gmail.com', 'password', 'templates')
    
    # Check if it returns a Flask app
    from flask import Flask
    assert isinstance(app, Flask), "Should return Flask instance"
    
    # Check if mail is configured
    assert 'MAIL_SERVER' in app.config, "MAIL_SERVER should be configured"
    assert app.config['MAIL_SERVER'] == 'smtp.gmail.com', "MAIL_SERVER incorrect"
    assert app.config['MAIL_PORT'] == 587, "MAIL_PORT incorrect"
    assert app.config['MAIL_USERNAME'] == 'user@gmail.com', "MAIL_USERNAME incorrect"
    assert app.config['MAIL_PASSWORD'] == 'password', "MAIL_PASSWORD incorrect"
    
    # Check if send route exists
    assert '/send' in [rule.rule for rule in app.url_map.iter_rules()], "Should have /send route"
    
    print("PASSED")
except Exception as e:
    print(f"FAILED: {e}")
"""

results = []
for i, solution in enumerate(solutions):
    print(f"\nEvaluating Solution {i+1}...")
    
    # Create test file
    full_code = solution + "\n\n" + TEST_CODE
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        passed = result.returncode == 0 and 'PASSED' in result.stdout
        results.append(1 if passed else 0)
        print(f"Result: {'PASS' if passed else 'FAIL'}")
        if not passed and result.stderr:
            print(f"Error: {result.stderr[:200]}...")
        
    except subprocess.TimeoutExpired:
        results.append(0)
        print("Result: FAIL (timeout)")
    except Exception as e:
        results.append(0)
        print(f"Result: FAIL ({e})")
    finally:
        os.unlink(temp_file)

# Step 4: Show pass@1 results
print("\n\n4. PASS@1 RESULTS:")
print("-"*40)
print(f"Results tuple: {tuple(results)}")
print(f"Pass rate: {sum(results)}/{len(results)} ({sum(results)/len(results)*100:.1f}%)")

# Show which strategies passed
print("\nStrategy performance:")
for i, (strategy, result) in enumerate(zip(strategies, results)):
    print(f"  {strategy['name']}: {'PASS' if result else 'FAIL'}") 