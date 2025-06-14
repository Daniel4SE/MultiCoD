"""Chain of Draft code generation"""

import re
import time
import logging
from typing import Dict, List, Tuple, Optional
import anthropic
from prompts.strategy_generation import COD_GENERATION_PROMPT

logger = logging.getLogger(__name__)

class CoDGenerator:
    """Generates code using Chain of Draft methodology"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def generate_code(self, task: Dict, strategy: Dict, 
                     temperature: float = 0.5) -> Dict:
        """Generate code using a specific CoD strategy"""
        
        # Build task description
        task_description = task.get('description', '')
        if 'docstring' in task:
            task_description = task['docstring']
        
        # Get function signature
        task_signature = task['prompt'].strip()
        
        prompt = COD_GENERATION_PROMPT.format(
            task_description=task_description,
            task_signature=task_signature,
            strategy_name=strategy['name'],
            strategy_focus=strategy['focus'],
            strategy_instruction=strategy['instruction']
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2500,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            
            # Extract token usage
            token_usage = {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            }
            
            # Parse response
            steps, code = self._parse_response(content)
            
            return {
                'success': True,
                'strategy': strategy['name'],
                'temperature': temperature,
                'steps': steps,
                'code': code,
                'raw_response': content,
                'token_usage': token_usage
            }
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return {
                'success': False,
                'strategy': strategy['name'],
                'temperature': temperature,
                'steps': [],
                'code': '',
                'error': str(e),
                'token_usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
            }
    
    def _parse_response(self, response: str) -> Tuple[List[str], str]:
        """Extract CoD steps and code from response"""
        steps = []
        code = ""
        
        # Extract draft steps
        draft_match = re.search(
            r'DRAFT:\s*\n(.*?)(?:SOLUTION:|```)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        
        if draft_match:
            draft_section = draft_match.group(1)
            # Find all steps
            step_pattern = r'step\d+:\s*(.+?)(?=step\d+:|SOLUTION:|```|$)'
            step_matches = re.findall(step_pattern, draft_section, re.DOTALL)
            steps = [step.strip() for step in step_matches if step.strip()]
        
        # Extract code
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        
        return steps, code
    
    def validate_steps(self, steps: List[str]) -> Dict:
        """Validate CoD steps adhere to rules"""
        validation = {
            'total_steps': len(steps),
            'valid_steps': 0,
            'violations': [],
            'avg_words': 0
        }
        
        word_counts = []
        for i, step in enumerate(steps):
            words = len(step.split())
            word_counts.append(words)
            
            if words <= 5:
                validation['valid_steps'] += 1
            else:
                validation['violations'].append({
                    'step': i + 1,
                    'content': step,
                    'word_count': words
                })
        
        if word_counts:
            validation['avg_words'] = sum(word_counts) / len(word_counts)
        
        validation['adherence_rate'] = (
            validation['valid_steps'] / validation['total_steps']
            if validation['total_steps'] > 0 else 0
        )
        
        return validation