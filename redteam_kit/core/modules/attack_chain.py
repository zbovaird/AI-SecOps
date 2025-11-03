"""
Multi-stage Attack Chain Execution
FOR AUTHORIZED SECURITY TESTING IN SANDBOXED ENVIRONMENTS ONLY

Usage for Red Teaming:
---------------------
The Attack Chain module orchestrates multi-stage attack workflows, executing
reconnaissance, exploitation, post-exploitation, persistence, and cleanup phases
in sequence. This simulates a complete red team engagement and helps test defensive
capabilities across the entire attack lifecycle.

Example Usage:
    from utils.logger import FrameworkLogger
    from core.modules.attack_chain import AttackChain, AttackStage
    
    # Initialize attack chain with target
    logger = FrameworkLogger("attack_chain_test")
    chain = AttackChain(logger, target="192.168.1.100")
    
    # Execute individual stage
    recon_result = chain.execute_stage(AttackStage.INITIAL_RECON)
    print(f"Recon stage: {recon_result['status']}")
    
    # Execute specific stages
    stages = [
        AttackStage.INITIAL_RECON,
        AttackStage.CREDENTIAL_HARVEST,
        AttackStage.PRIVILEGE_ESCALATION,
        AttackStage.PERSISTENCE
    ]
    
    # Execute full attack chain
    results = chain.execute_full_chain(stages)
    print(f"Attack chain status: {results['overall_status']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    print(f"Stages completed: {len(results['stages'])}")
    
    # Get all results
    all_results = chain.get_results()
    print(f"Chain complete: {all_results['chain_complete']}")
    
    # View stage results
    for stage_name, stage_result in all_results['stage_results'].items():
        print(f"{stage_name}: {stage_result['status']}")

Red Team Use Cases:
- Simulating complete attack workflows
- Testing defensive detection across attack lifecycle
- Multi-stage engagement orchestration
- Reconnaissance phase execution
- Post-exploitation activities
- Persistence establishment
- Lateral movement testing
- Data collection and exfiltration
- Cleanup phase execution
- Full engagement simulation
"""

import time
import random
from typing import Dict, List, Optional, Callable
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import FrameworkLogger
# Lazy imports to reduce memory footprint - modules imported only when needed


class AttackStage(Enum):
    """Attack chain stages"""
    INITIAL_RECON = "initial_recon"
    CREDENTIAL_HARVEST = "credential_harvest"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PERSISTENCE = "persistence"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_COLLECTION = "data_collection"
    DATA_EXFILTRATION = "data_exfiltration"
    CLEANUP = "cleanup"


class AttackChain:
    """Multi-stage attack chain orchestrator"""
    
    def __init__(self, logger: Optional[FrameworkLogger] = None, target: Optional[str] = None):
        """
        Initialize attack chain
        
        Args:
            logger: Logger instance (optional)
            target: Target IP address, domain, or URL (optional)
        """
        self.logger = logger or FrameworkLogger("attack_chain")
        self.target = target
        # Lazy initialization - modules created only when needed to reduce memory
        self._recon = None
        self._post_exploit = None
        self._exploit = None
        self._persistence = None
        self._evasion = None
        self.stage_results: Dict[str, Dict] = {}
        self.chain_complete = False
    
    @property
    def recon(self):
        """Lazy load recon module"""
        if self._recon is None:
            from core.modules.recon import ReconModule
            self._recon = ReconModule(self.logger, target=self.target)
        return self._recon
    
    @property
    def post_exploit(self):
        """Lazy load post_exploit module"""
        if self._post_exploit is None:
            from core.modules.post_exploit import PostExploitation
            self._post_exploit = PostExploitation(self.logger, target=self.target)
        return self._post_exploit
    
    @property
    def exploit(self):
        """Lazy load exploit module"""
        if self._exploit is None:
            from core.modules.exploit import ExploitModule
            self._exploit = ExploitModule(self.logger)
        return self._exploit
    
    @property
    def persistence(self):
        """Lazy load persistence module"""
        if self._persistence is None:
            from core.modules.advanced_persistence import AdvancedPersistence
            self._persistence = AdvancedPersistence(self.logger)
        return self._persistence
    
    @property
    def evasion(self):
        """Lazy load evasion module"""
        if self._evasion is None:
            from core.modules.advanced_evasion import AdvancedEvasionModule
            self._evasion = AdvancedEvasionModule(self.target, self.logger)
        return self._evasion
    
    def execute_stage(self, stage: AttackStage, delay: float = None) -> Dict:
        """
        Execute a specific attack stage
        
        Args:
            stage: Attack stage to execute
            delay: Optional delay before execution
        
        Returns:
            Stage execution results
        """
        if delay:
            time.sleep(delay)
        
        self.logger.info(f"Executing stage: {stage.value}")
        
        stage_results = {
            "stage": stage.value,
            "status": "pending",
            "results": {}
        }
        
        try:
            if stage == AttackStage.INITIAL_RECON:
                stage_results["results"] = self.recon.perform_recon(target=self.target)
            elif stage == AttackStage.CREDENTIAL_HARVEST:
                stage_results["results"] = self.post_exploit.harvest_credentials()
            elif stage == AttackStage.PRIVILEGE_ESCALATION:
                stage_results["results"] = self.post_exploit.escalate_privileges()
            elif stage == AttackStage.PERSISTENCE:
                stage_results["results"] = self.persistence.establish_persistence()
            elif stage == AttackStage.LATERAL_MOVEMENT:
                stage_results["results"] = self.post_exploit.move_laterally()
            elif stage == AttackStage.DATA_COLLECTION:
                stage_results["results"] = self.post_exploit.collect_data()
            elif stage == AttackStage.DATA_EXFILTRATION:
                stage_results["results"] = self.post_exploit.exfiltrate_data()
            elif stage == AttackStage.CLEANUP:
                stage_results["results"] = self._cleanup()
            
            stage_results["status"] = "completed"
            
        except Exception as e:
            self.logger.error(f"Error executing stage {stage.value}: {e}")
            stage_results["status"] = "failed"
            stage_results["error"] = str(e)
        
        self.stage_results[stage.value] = stage_results
        return stage_results
    
    def execute_full_chain(self, stages: List[AttackStage] = None) -> Dict:
        """
        Execute full attack chain
        
        Args:
            stages: Optional list of stages (defaults to all)
        
        Returns:
            Complete chain results
        """
        if stages is None:
            stages = list(AttackStage)
        
        self.logger.info("Starting full attack chain execution")
        
        chain_results = {
            "start_time": time.time(),
            "stages": [],
            "overall_status": "in_progress"
        }
        
        for stage in stages:
            if stage == AttackStage.CLEANUP:
                # Always execute cleanup last
                continue
            
            result = self.execute_stage(stage, delay=random.uniform(0.5, 2.0))
            chain_results["stages"].append(result)
        
        # Execute cleanup
        cleanup_result = self.execute_stage(AttackStage.CLEANUP)
        chain_results["stages"].append(cleanup_result)
        
        chain_results["end_time"] = time.time()
        chain_results["duration"] = chain_results["end_time"] - chain_results["start_time"]
        chain_results["overall_status"] = "completed"
        self.chain_complete = True
        
        self.logger.info(f"Attack chain completed in {chain_results['duration']:.2f} seconds")
        
        return chain_results
    
    def _cleanup(self) -> Dict:
        """Cleanup operations"""
        self.logger.info("Performing cleanup")
        return {
            "artifacts_removed": 0,
            "logs_cleared": True,
            "status": "cleanup_complete"
        }
    
    def get_results(self) -> Dict:
        """Get all stage results"""
        return {
            "stage_results": self.stage_results,
            "chain_complete": self.chain_complete
        }

