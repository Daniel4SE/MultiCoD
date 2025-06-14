"""BigCodeBench dataset loader"""

import logging
from typing import List, Dict, Optional
from datasets import load_dataset

logger = logging.getLogger(__name__)

class BigCodeBenchLoader:
    """Loads and manages BigCodeBench dataset"""
    
    def __init__(self, subset: str = "complete", cache_dir: str = "./cache", version: str = "v0.4.1"):
        self.subset = subset
        self.cache_dir = cache_dir
        self.version = version
        self.dataset = None
    
    def load_dataset(self, max_tasks: Optional[int] = None) -> List[Dict]:
        """Load BigCodeBench dataset"""
        logger.info(f"Loading BigCodeBench {self.subset} subset version {self.version}...")
        
        try:
            # Load from HuggingFace
            dataset = load_dataset(
                "bigcode/bigcodebench",
                None,  # Use default configuration
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Check available splits
            if self.version not in dataset:
                logger.warning(f"Version {self.version} not found. Available: {list(dataset.keys())}")
                # Try to use the latest available version
                if 'v0.1.4' in dataset:
                    self.version = 'v0.1.4'
                    logger.info(f"Using available version: {self.version}")
                else:
                    # Use the first available split
                    self.version = list(dataset.keys())[0]
                    logger.info(f"Using first available version: {self.version}")
            
            # Convert to list of dicts
            tasks = []
            for item in dataset[self.version]:
                # Parse doc_struct if it's a string
                description = ''
                if item.get('doc_struct'):
                    try:
                        import json
                        doc_struct = json.loads(item['doc_struct']) if isinstance(item['doc_struct'], str) else item['doc_struct']
                        if isinstance(doc_struct, dict) and 'description' in doc_struct:
                            desc = doc_struct['description']
                            description = desc if isinstance(desc, str) else ' '.join(desc) if isinstance(desc, list) else ''
                    except:
                        description = ''
                
                # Handle different field names across versions
                prompt_field = 'complete_prompt' if 'complete_prompt' in item else 'prompt'
                
                task = {
                    'task_id': item['task_id'],
                    'prompt': item[prompt_field],
                    'canonical_solution': item['canonical_solution'],
                    'test': item.get('test', ''),
                    'entry_point': item.get('entry_point', 'task_func'),
                    'description': description,
                    'docstring': self._extract_docstring(item[prompt_field])
                }
                tasks.append(task)
            
            # Limit tasks if specified
            if max_tasks:
                tasks = tasks[:max_tasks]
            
            logger.info(f"Loaded {len(tasks)} tasks from {self.version}")
            return tasks
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Return sample task for testing
            return self._get_sample_tasks()
    
    def _extract_docstring(self, prompt: str) -> str:
        """Extract docstring from prompt"""
        import re
        match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"'''(.*?)'''", prompt, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _get_sample_tasks(self) -> List[Dict]:
        """Get sample tasks for testing"""
        return [
            {
                'task_id': 'sample/001',
                'prompt': '''from flask import Flask
from flask_mail import Mail, Message
def task_func(smtp_server, smtp_port, smtp_user, smtp_password, template_folder):
    """
    Creates a Flask application configured to send emails using Flask-Mail.
    """''',
                'canonical_solution': '''
    app = Flask(__name__, template_folder=template_folder)
    app.config['MAIL_SERVER'] = smtp_server
    app.config['MAIL_PORT'] = smtp_port
    app.config['MAIL_USERNAME'] = smtp_user
    app.config['MAIL_PASSWORD'] = smtp_password
    app.config['MAIL_USE_TLS'] = True
    
    mail = Mail()
    mail.init_app(app)
    
    @app.route('/send_mail')
    def send_mail():
        msg = Message('Test', sender='from@example.com', recipients=['to@example.com'])
        msg.body = 'Test email'
        mail.send(msg)
        return 'Sent'
    
    return app''',
                'test': '',
                'entry_point': 'task_func',
                'description': 'Create Flask app with email functionality',
                'docstring': 'Creates a Flask application configured to send emails using Flask-Mail.'
            }
        ]