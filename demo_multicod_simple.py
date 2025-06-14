#!/usr/bin/env python3
"""
Simplified Multi-CoD Demonstration showing generated code
"""

print("="*60)
print("MULTI-COD DEMONSTRATION: Flask Email Task")
print("="*60)

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

print("\nINITIAL PROMPT:")
print(INITIAL_PROMPT)

# Step 1: Show 5 Chain-of-Draft strategies
print("\n\n1. GENERATED 5 CHAIN-OF-DRAFT STRATEGIES:")
print("-"*40)

strategies = [
    {
        "name": "Configuration-First Approach",
        "focus": "Set up all configs before creating mail instance",
        "draft_steps": [
            "Create Flask app",
            "Set mail configurations",
            "Initialize Mail instance",
            "Define send route",
            "Return configured app"
        ]
    },
    {
        "name": "Route-Centric Strategy",
        "focus": "Define routes first, then configure mail",
        "draft_steps": [
            "Initialize Flask app",
            "Create send route decorator",
            "Configure mail settings",
            "Setup Mail instance",
            "Return complete app"
        ]
    },
    {
        "name": "Error-Handling Focus",
        "focus": "Include error handling for SMTP failures",
        "draft_steps": [
            "Create Flask instance",
            "Add error handlers",
            "Configure SMTP settings",
            "Create mail sender",
            "Return safe app"
        ]
    },
    {
        "name": "Template-Aware Design",
        "focus": "Properly handle template folder configuration",
        "draft_steps": [
            "Setup Flask with templates",
            "Configure mail server",
            "Initialize Mail object",
            "Add email route",
            "Return configured Flask"
        ]
    },
    {
        "name": "Minimal Implementation",
        "focus": "Simplest working solution",
        "draft_steps": [
            "Create app instance",
            "Set config values",
            "Init Mail extension",
            "Add send endpoint",
            "Return Flask app"
        ]
    }
]

for i, strategy in enumerate(strategies):
    print(f"\nStrategy {i+1}: {strategy['name']}")
    print(f"Focus: {strategy['focus']}")
    print("Draft steps:")
    for j, step in enumerate(strategy['draft_steps']):
        print(f"  step{j+1}: {step}")

# Step 2: Show generated solutions
print("\n\n2. GENERATED SOLUTIONS:")
print("-"*40)

solutions = [
    # Solution 1: Configuration-First
    """from flask import Flask
from flask_mail import Mail, Message

def task_func(smtp_server, smtp_port, smtp_user, smtp_password, template_folder):
    app = Flask(__name__, template_folder=template_folder)
    
    app.config['MAIL_SERVER'] = smtp_server
    app.config['MAIL_PORT'] = smtp_port
    app.config['MAIL_USERNAME'] = smtp_user
    app.config['MAIL_PASSWORD'] = smtp_password
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USE_SSL'] = False
    
    mail = Mail(app)
    
    @app.route('/send')
    def send_email():
        msg = Message('Test Email', sender=smtp_user, recipients=['test@example.com'])
        msg.body = 'This is a test email'
        mail.send(msg)
        return 'Email sent!'
    
    return app""",
    
    # Solution 2: Route-Centric
    """from flask import Flask
from flask_mail import Mail, Message

def task_func(smtp_server, smtp_port, smtp_user, smtp_password, template_folder):
    app = Flask(__name__, template_folder=template_folder)
    
    @app.route('/send')
    def send_test_email():
        msg = Message('Test', sender=smtp_user, recipients=['recipient@example.com'])
        msg.body = 'Test email body'
        mail.send(msg)
        return 'Sent'
    
    app.config.update(
        MAIL_SERVER=smtp_server,
        MAIL_PORT=smtp_port,
        MAIL_USERNAME=smtp_user,
        MAIL_PASSWORD=smtp_password,
        MAIL_USE_TLS=True
    )
    
    mail = Mail(app)
    return app""",
    
    # Solution 3: Error-Handling (has bug - missing return)
    """from flask import Flask
from flask_mail import Mail, Message

def task_func(smtp_server, smtp_port, smtp_user, smtp_password, template_folder):
    app = Flask(__name__, template_folder=template_folder)
    
    @app.errorhandler(Exception)
    def handle_error(e):
        return str(e), 500
    
    app.config['MAIL_SERVER'] = smtp_server
    app.config['MAIL_PORT'] = smtp_port
    app.config['MAIL_USERNAME'] = smtp_user
    app.config['MAIL_PASSWORD'] = smtp_password
    
    mail = Mail(app)
    
    @app.route('/send')
    def send():
        try:
            msg = Message('Test', sender=smtp_user, recipients=['test@test.com'])
            mail.send(msg)
            return 'Sent successfully'
        except Exception as e:
            return f'Failed: {e}'""",  # Missing return statement!
    
    # Solution 4: Template-Aware
    """from flask import Flask
from flask_mail import Mail, Message

def task_func(smtp_server, smtp_port, smtp_user, smtp_password, template_folder):
    app = Flask(__name__, template_folder=template_folder)
    
    app.config['MAIL_SERVER'] = smtp_server
    app.config['MAIL_PORT'] = smtp_port
    app.config['MAIL_USERNAME'] = smtp_user
    app.config['MAIL_PASSWORD'] = smtp_password
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_DEFAULT_SENDER'] = smtp_user
    
    mail = Mail()
    mail.init_app(app)
    
    @app.route('/send')
    def send_email():
        message = Message(
            subject='Test Email',
            recipients=['recipient@example.com'],
            body='This is a test email from Flask-Mail'
        )
        mail.send(message)
        return 'Email sent successfully!'
    
    return app""",
    
    # Solution 5: Minimal (has bug - mail used before initialization)
    """from flask import Flask
from flask_mail import Mail, Message

def task_func(smtp_server, smtp_port, smtp_user, smtp_password, template_folder):
    app = Flask(__name__)
    
    app.config['MAIL_SERVER'] = smtp_server
    app.config['MAIL_PORT'] = smtp_port
    app.config['MAIL_USERNAME'] = smtp_user
    app.config['MAIL_PASSWORD'] = smtp_password
    
    @app.route('/send')
    def send():
        msg = Message('Test', sender=smtp_user, recipients=['test@example.com'])
        mail.send(msg)  # Bug: mail not initialized yet!
        return 'Sent'
    
    mail = Mail(app)
    return app"""
]

for i, (strategy, solution) in enumerate(zip(strategies, solutions)):
    print(f"\nSolution {i+1} ({strategy['name']}):")
    print("```python")
    print(solution)
    print("```")

# Step 3: Simulate evaluation
print("\n\n3. EVALUATION RESULTS:")
print("-"*40)

# Simulate evaluation based on common patterns/bugs
evaluation_results = []

for i, solution in enumerate(solutions):
    checks = {
        "has_return": "return app" in solution,
        "mail_initialized": "mail = Mail" in solution or "mail.init_app" in solution,
        "route_defined": "@app.route('/send')" in solution,
        "config_set": all(config in solution for config in ['MAIL_SERVER', 'MAIL_PORT', 'MAIL_USERNAME', 'MAIL_PASSWORD']),
        "template_folder": "template_folder=template_folder" in solution,
        "no_syntax_error": True  # Simplified - in reality would parse
    }
    
    # Special cases for known bugs
    if i == 2:  # Solution 3 missing return
        checks["has_return"] = False
    if i == 4:  # Solution 5 has order bug
        checks["no_syntax_error"] = False  # Would fail at runtime
    
    passed = all(checks.values())
    evaluation_results.append(1 if passed else 0)
    
    print(f"\nSolution {i+1} evaluation:")
    for check, result in checks.items():
        print(f"  {check}: {'✓' if result else '✗'}")
    print(f"  Overall: {'PASS' if passed else 'FAIL'}")

# Step 4: Show pass@1 results
print("\n\n4. PASS@1 RESULTS:")
print("-"*40)
print(f"Results tuple: {tuple(evaluation_results)}")
print(f"Example: (1, 1, 0, 1, 0) means solutions 1,2,4 passed, 3,5 failed")
print(f"\nIn this run: {sum(evaluation_results)}/5 solutions passed")
print("\nThis demonstrates how Multi-CoD generates diverse solutions,")
print("increasing the chance that at least one will pass the tests!") 