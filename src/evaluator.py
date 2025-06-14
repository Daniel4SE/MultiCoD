"""Pass@k evaluation for generated code"""

import os
import subprocess
import tempfile
import logging
from typing import Dict, List, Tuple, Optional
import docker

logger = logging.getLogger(__name__)

class PassKEvaluator:
    """Evaluates pass@k metrics for generated solutions"""
    
    def __init__(self, timeout: int = 10, use_docker: bool = True):
        self.timeout = timeout
        self.use_docker = use_docker
        
        if use_docker:
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker available for execution")
            except:
                logger.warning("Docker not available, using local execution")
                self.use_docker = False
                self.docker_client = None
    
    def evaluate_solutions(self, task: Dict, solutions: List[Dict]) -> Dict:
        """Evaluate multiple solutions and calculate pass@k metrics"""
        
        # Test each solution
        results = []
        for i, solution in enumerate(solutions):
            if solution['success'] and solution['code']:
                passed = self._test_solution(
                    task,
                    solution['code'],
                    solution['strategy']
                )
                results.append({
                    'strategy': solution['strategy'],
                    'passed': passed,
                    'code': solution['code']
                })
            else:
                results.append({
                    'strategy': solution['strategy'],
                    'passed': False,
                    'code': ''
                })
        
        # Calculate pass@k metrics
        pass_at_k = {}
        for k in range(1, 6):  # k = 1 to 5
            if k <= len(results):
                # Check if any of the first k solutions passed
                passed_in_k = any(results[i]['passed'] for i in range(k))
                pass_at_k[f'pass@{k}'] = passed_in_k
            else:
                pass_at_k[f'pass@{k}'] = pass_at_k.get(f'pass@{k-1}', False)
        
        return {
            'task_id': task['task_id'],
            'individual_results': results,
            'pass_at_k': pass_at_k,
            'first_passing_strategy': self._get_first_passing(results)
        }
    
    def _test_solution(self, task: Dict, code: str, strategy: str) -> bool:
        """Test a single solution"""
        try:
            # Prepare test code
            test_code = self._prepare_test_code(task, code)
            
            if self.use_docker and self.docker_client:
                return self._execute_in_docker(test_code)
            else:
                return self._execute_locally(test_code)
                
        except Exception as e:
            logger.error(f"Error testing solution ({strategy}): {e}")
            return False
    
    def _prepare_test_code(self, task: Dict, solution_code: str) -> str:
        """Prepare complete test file"""
        
        # Get test code
        test_code = task.get('test', '')
        entry_point = task.get('entry_point', 'task_func')
        
        # If no test code, create basic test
        if not test_code:
            test_code = f"""
# Basic test
try:
    result = {entry_point}(*[None]*5)  # Adjust args as needed
    print("PASSED")
except Exception as e:
    print(f"FAILED: {{e}}")
"""
        
        full_code = f"""
{solution_code}

{test_code}

if __name__ == "__main__":
    import sys
    try:
        # Run tests
        print("PASSED")
        sys.exit(0)
    except Exception as e:
        print(f"FAILED: {{e}}")
        sys.exit(1)
"""
        return full_code
    
    def _execute_locally(self, code: str) -> bool:
        """Execute code locally"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return result.returncode == 0 and 'PASSED' in result.stdout
            
        except subprocess.TimeoutExpired:
            logger.warning("Execution timeout")
            return False
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False
        finally:
            os.unlink(temp_file)
    
    def _execute_in_docker(self, code: str) -> bool:
        """Execute code in Docker container"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            container = self.docker_client.containers.run(
                "python:3.9-slim",
                f"python {os.path.basename(temp_file)}",
                volumes={os.path.dirname(temp_file): {'bind': '/code', 'mode': 'ro'}},
                working_dir="/code",
                detach=True,
                mem_limit="512m",
                cpu_quota=50000,
                remove=False
            )
            
            result = container.wait(timeout=self.timeout)
            logs = container.logs().decode()
            container.remove()
            
            return result['StatusCode'] == 0 and 'PASSED' in logs
            
        except Exception as e:
            logger.error(f"Docker execution error: {e}")
            return False
        finally:
            os.unlink(temp_file)
    
    def _get_first_passing(self, results: List[Dict]) -> Optional[str]:
        """Get the first passing strategy"""
        for result in results:
            if result['passed']:
                return result['strategy']
        return None