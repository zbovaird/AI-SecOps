"""
PyRIT Integration Module
Integrates Microsoft PyRIT capabilities with redteam_kit

Usage:
    from redteam_kit.core.modules.pyrit_integration import PyRITOrchestrator
    
    orchestrator = PyRITOrchestrator()
    results = orchestrator.run_comprehensive_test(target_model, test_prompts)
"""

from typing import List, Dict, Optional, Any, Callable, TYPE_CHECKING
import json
from datetime import datetime

# Conditional imports for type checking only
if TYPE_CHECKING:
    from pyrit.memory import MemoryInterface

try:
    import pyrit
    # Try importing core components
    from pyrit.models import PromptRequestPiece
    from pyrit.memory import DuckDBMemory, MemoryInterface
    from pyrit.score import (
        SelfAskGeneralScorer,
        SelfAskRefusalScorer,
        SelfAskCategoryScorer,
        CompositeScorer
    )
    from pyrit.datasets import fetch_harmbench_dataset
    # Orchestrators
    try:
        from pyrit.orchestrator import MultiTurnOrchestrator, PromptSendingOrchestrator
        ORCHESTRATORS_AVAILABLE = True
    except ImportError:
        ORCHESTRATORS_AVAILABLE = False
        MultiTurnOrchestrator = None
        PromptSendingOrchestrator = None
    # Attack Strategies
    try:
        from pyrit.strategy import (
            Base64Attack,
            CaesarAttack,
            AtbashAttack,
            CharacterSpaceAttack,
            CharSwapAttack
        )
        ATTACK_STRATEGIES_AVAILABLE = True
    except ImportError:
        ATTACK_STRATEGIES_AVAILABLE = False
        Base64Attack = None
        CaesarAttack = None
        AtbashAttack = None
        CharacterSpaceAttack = None
        CharSwapAttack = None
    PYRIT_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    PYRIT_AVAILABLE = False
    ORCHESTRATORS_AVAILABLE = False
    ATTACK_STRATEGIES_AVAILABLE = False
    # Define placeholder types when PyRIT is not available
    MemoryInterface = Any
    DuckDBMemory = None
    PromptRequestPiece = Any
    MultiTurnOrchestrator = None
    PromptSendingOrchestrator = None
    Base64Attack = None
    CaesarAttack = None
    AtbashAttack = None
    CharacterSpaceAttack = None
    CharSwapAttack = None
    # Only print warning if this is the first import attempt (not a reload)
    import sys
    if 'pyrit' not in sys.modules:
        print("Warning: PyRIT not available. Install with: pip install pyrit")


class PyRITOrchestrator:
    """
    Orchestrates PyRIT capabilities for comprehensive red teaming
    """
    
    def __init__(
        self,
        memory: Optional[Any] = None,  # Use Any instead of MemoryInterface to avoid NameError
        enable_scoring: bool = True
    ):
        """
        Initialize PyRIT orchestrator
        
        Args:
            memory: PyRIT memory interface (creates DuckDBMemory if None)
            enable_scoring: Whether to enable automatic scoring
        """
        if not PYRIT_AVAILABLE:
            raise ImportError(
                "PyRIT is required. Install with: pip install pyrit"
            )
        
        # DuckDBMemory is already imported at module level when PyRIT is available
        self.memory = memory or DuckDBMemory()
        self.enable_scoring = enable_scoring
        
        # Initialize scorers
        if enable_scoring:
            self.general_scorer = SelfAskGeneralScorer()
            self.refusal_scorer = SelfAskRefusalScorer()
            self.category_scorer = SelfAskCategoryScorer()
            
            # Composite scorer for multiple metrics
            self.composite_scorer = CompositeScorer(
                scorers=[self.general_scorer, self.refusal_scorer]
            )
        else:
            self.general_scorer = None
            self.refusal_scorer = None
            self.category_scorer = None
            self.composite_scorer = None
    
    def test_prompt(
        self,
        prompt: str,
        target_model: Any,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test a single prompt using PyRIT framework
        
        Args:
            prompt: Prompt to test
            target_model: Model or API target to test against
            conversation_id: Optional conversation ID for tracking
        
        Returns:
            Dictionary with test results and scores
        """
        # Create prompt request
        request = PromptRequestPiece(
            role="user",
            original_value=prompt,
            conversation_id=conversation_id or f"test_{datetime.now().isoformat()}"
        )
        
        # Send to target model
        try:
            if hasattr(target_model, 'send_prompt'):
                response = target_model.send_prompt(prompt)
            elif callable(target_model):
                response = target_model(prompt)
            else:
                raise ValueError("target_model must be callable or have send_prompt method")
            
            # Store in memory
            self.memory.add_request_response_to_memory(
                request=request,
                response=response
            )
            
            # Score if enabled
            scores = {}
            if self.enable_scoring:
                # General score
                general_score = self.general_scorer.score_text(
                    text=response.completed_prompt_text if hasattr(response, 'completed_prompt_text') else str(response)
                )
                scores['general'] = general_score.score_value if hasattr(general_score, 'score_value') else None
                
                # Refusal score
                refusal_score = self.refusal_scorer.score_text(
                    text=response.completed_prompt_text if hasattr(response, 'completed_prompt_text') else str(response)
                )
                scores['refusal'] = refusal_score.score_value if hasattr(refusal_score, 'score_value') else None
            
            return {
                'prompt': prompt,
                'response': str(response),
                'scores': scores,
                'conversation_id': request.conversation_id,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
        
        except Exception as e:
            return {
                'prompt': prompt,
                'response': None,
                'error': str(e),
                'conversation_id': request.conversation_id,
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
    
    def test_prompts_batch(
        self,
        prompts: List[str],
        target_model: Any,
        conversation_id_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Test multiple prompts in batch
        
        Args:
            prompts: List of prompts to test
            target_model: Model or API target
            conversation_id_prefix: Prefix for conversation IDs
        
        Returns:
            List of test results
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            conv_id = f"{conversation_id_prefix}_{i}" if conversation_id_prefix else None
            result = self.test_prompt(prompt, target_model, conv_id)
            results.append(result)
        
        return results
    
    def load_harmbench_dataset(self) -> List[str]:
        """
        Load HarmBench dataset from PyRIT
        
        Returns:
            List of prompts from HarmBench
        """
        try:
            dataset = fetch_harmbench_dataset()
            prompts = []
            
            # Extract prompts from dataset
            if hasattr(dataset, 'prompts'):
                prompts = dataset.prompts
            elif isinstance(dataset, list):
                prompts = dataset
            elif hasattr(dataset, '__iter__'):
                prompts = list(dataset)
            
            return prompts
        
        except Exception as e:
            print(f"Warning: Could not load HarmBench dataset: {e}")
            return []
    
    def detect_refusal(
        self,
        response: str
    ) -> Dict[str, Any]:
        """
        Detect if response is a refusal
        
        Args:
            response: Model response text
        
        Returns:
            Dictionary with refusal detection results
        """
        if not self.refusal_scorer:
            return {'refusal_detected': False, 'score': None, 'error': 'Scorer not initialized'}
        
        try:
            score = self.refusal_scorer.score_text(text=response)
            refusal_detected = (
                hasattr(score, 'score_value') and score.score_value > 0.5
            ) or (
                hasattr(score, 'get_value') and score.get_value() > 0.5
            )
            
            return {
                'refusal_detected': refusal_detected,
                'score': score.score_value if hasattr(score, 'score_value') else None,
                'response': response
            }
        
        except Exception as e:
            return {
                'refusal_detected': False,
                'score': None,
                'error': str(e)
            }
    
    def export_results(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        format: str = 'json'
    ):
        """
        Export test results to file
        
        Args:
            results: List of test results
            output_path: Path to save results
            format: Format ('json', 'csv')
        """
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif format == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of PyRIT memory
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            # Get all conversations
            conversations = self.memory.get_all_prompt_pieces()
            
            return {
                'total_prompts': len(conversations) if conversations else 0,
                'memory_type': type(self.memory).__name__,
                'has_scoring': self.enable_scoring
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'memory_type': type(self.memory).__name__
            }
    
    def create_multi_turn_attack(
        self,
        initial_prompt: str,
        adversarial_model: Any,
        target_model: Any,
        max_turns: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Create multi-turn attack using MultiTurnOrchestrator
        
        Args:
            initial_prompt: Starting prompt for the attack
            adversarial_model: Model used to generate adversarial prompts
            target_model: Target model being attacked
            max_turns: Maximum number of conversation turns
            
        Returns:
            List of attack results from each turn
        """
        if not ORCHESTRATORS_AVAILABLE or MultiTurnOrchestrator is None:
            print("Warning: MultiTurnOrchestrator not available. Install PyRIT with orchestrator support.")
            return []
        
        try:
            orchestrator = MultiTurnOrchestrator(
                attack_content=initial_prompt,
                adversarial_model=adversarial_model,
                target_model=target_model,
                memory=self.memory,
                max_turns=max_turns
            )
            
            results = orchestrator.send_all_attacks()
            
            # Convert results to our format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'prompt': result.get('prompt', ''),
                    'response': result.get('response', ''),
                    'turn': result.get('turn', 0),
                    'success': result.get('success', False),
                    'timestamp': datetime.now().isoformat()
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in multi-turn attack: {e}")
            return []
    
    def send_prompts_async(
        self,
        prompts: List[str],
        target_model: Any,
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Send multiple prompts asynchronously using PromptSendingOrchestrator
        
        Args:
            prompts: List of prompts to send
            target_model: Target model
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of results
        """
        if not ORCHESTRATORS_AVAILABLE or PromptSendingOrchestrator is None:
            # Fallback to sequential processing
            return self.test_prompts_batch(prompts, target_model)
        
        try:
            orchestrator = PromptSendingOrchestrator(
                target_model=target_model,
                memory=self.memory,
                max_concurrent=max_concurrent
            )
            
            # Create prompt requests
            requests = [
                PromptRequestPiece(
                    role="user",
                    original_value=prompt,
                    conversation_id=f"async_{i}"
                )
                for i, prompt in enumerate(prompts)
            ]
            
            results = orchestrator.send_prompts(requests)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'prompt': result.get('prompt', ''),
                    'response': result.get('response', ''),
                    'success': result.get('success', False),
                    'timestamp': datetime.now().isoformat()
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in async prompt sending: {e}")
            # Fallback to sequential
            return self.test_prompts_batch(prompts, target_model)
    
    def apply_encoding_attack(
        self,
        prompt: str,
        strategy: str = "base64"
    ) -> str:
        """
        Apply encoding-based attack strategy to a prompt
        
        Args:
            prompt: Original prompt
            strategy: Attack strategy ('base64', 'caesar', 'atbash', 'charspace', 'charswap')
            
        Returns:
            Transformed prompt
        """
        if not ATTACK_STRATEGIES_AVAILABLE:
            print(f"Warning: Attack strategies not available. Returning original prompt.")
            return prompt
        
        try:
            if strategy == "base64" and Base64Attack:
                attack = Base64Attack()
                return attack.transform(prompt)
            elif strategy == "caesar" and CaesarAttack:
                attack = CaesarAttack()
                return attack.transform(prompt)
            elif strategy == "atbash" and AtbashAttack:
                attack = AtbashAttack()
                return attack.transform(prompt)
            elif strategy == "charspace" and CharacterSpaceAttack:
                attack = CharacterSpaceAttack()
                return attack.transform(prompt)
            elif strategy == "charswap" and CharSwapAttack:
                attack = CharSwapAttack()
                return attack.transform(prompt)
            else:
                print(f"Warning: Strategy '{strategy}' not available. Returning original prompt.")
                return prompt
                
        except Exception as e:
            print(f"Error applying {strategy} attack: {e}")
            return prompt
    
    def apply_all_encoding_attacks(
        self,
        prompt: str
    ) -> Dict[str, str]:
        """
        Apply all available encoding attacks to a prompt
        
        Args:
            prompt: Original prompt
            
        Returns:
            Dictionary mapping strategy names to transformed prompts
        """
        strategies = ["base64", "caesar", "atbash", "charspace", "charswap"]
        results = {}
        
        for strategy in strategies:
            transformed = self.apply_encoding_attack(prompt, strategy)
            if transformed != prompt:  # Only include if transformation occurred
                results[strategy] = transformed
        
        return results
    
    def test_with_encoding_attacks(
        self,
        prompt: str,
        target_model: Any,
        strategies: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Test a prompt with multiple encoding attack strategies
        
        Args:
            prompt: Original prompt
            target_model: Target model
            strategies: List of strategies to test (None = all available)
            
        Returns:
            List of test results for each strategy
        """
        if strategies is None:
            strategies = ["base64", "caesar", "atbash", "charspace", "charswap"]
        
        results = []
        
        # Test original prompt first
        original_result = self.test_prompt(prompt, target_model, "encoding_original")
        original_result['strategy'] = 'original'
        results.append(original_result)
        
        # Test each encoding strategy
        for strategy in strategies:
            transformed_prompt = self.apply_encoding_attack(prompt, strategy)
            if transformed_prompt != prompt:
                result = self.test_prompt(transformed_prompt, target_model, f"encoding_{strategy}")
                result['strategy'] = strategy
                result['original_prompt'] = prompt
                result['transformed_prompt'] = transformed_prompt
                results.append(result)
        
        return results
