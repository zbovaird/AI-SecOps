"""
Multi-stage Attack Chain Execution
FOR AUTHORIZED SECURITY TESTING IN SANDBOXED ENVIRONMENTS ONLY
"""

import time
import random
from typing import Dict, List, Optional, Callable
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import FrameworkLogger
from core.modules.recon import ReconModule
from core.modules.post_exploit import PostExploitation
from core.modules.exploit import ExploitModule
from core.modules.advanced_persistence import AdvancedPersistence
from core.modules.advanced_evasion import AdvancedEvasionModule


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
    
    def __init__(self, logger: Optional[FrameworkLogger] = None):
        """Initialize attack chain"""
        self.logger = logger or FrameworkLogger("attack_chain")
        self.recon = ReconModule(self.logger)
        self.post_exploit = PostExploitation(self.logger)
        self.exploit = ExploitModule(self.logger)
        self.persistence = AdvancedPersistence(self.logger)
        self.evasion = AdvancedEvasionModule(None, self.logger)
        self.stage_results: Dict[str, Dict] = {}
        self.chain_complete = False
    
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
                stage_results["results"] = self.recon.perform_recon()
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

