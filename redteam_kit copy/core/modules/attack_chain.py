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
    
    # Use predefined attack profiles
    recon_only = chain.execute_full_chain(profile="recon_only")
    initial_access = chain.execute_full_chain(profile="initial_access")
    post_exploit = chain.execute_full_chain(profile="post_exploit_focus")
    
    # Execute specific stages
    stages = [
        AttackStage.OS_DETECTION,
        AttackStage.INITIAL_RECON,
        AttackStage.CREDENTIAL_HARVEST,
        AttackStage.PRIVILEGE_ESCALATION
    ]
    results = chain.execute_full_chain(stages=stages)
    
    # Use profile but exclude cleanup
    results = chain.execute_full_chain(profile="full_engagement", exclude_stages=[AttackStage.CLEANUP])
    
    # Create custom profile
    chain.create_custom_profile("my_custom", [
        AttackStage.OS_DETECTION,
        AttackStage.INITIAL_RECON,
        AttackStage.CREDENTIAL_HARVEST
    ])
    results = chain.execute_full_chain(profile="my_custom")
    
    # View available profiles
    profiles = chain.get_available_profiles()
    print(f"Available profiles: {list(profiles.keys())}")
    
    # Get all results
    all_results = chain.get_results()
    print(f"Chain complete: {all_results['chain_complete']}")

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
    OS_DETECTION = "os_detection"
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
    
    # Attack profiles - predefined module combinations
    ATTACK_PROFILES = {
        "recon_only": [
            AttackStage.OS_DETECTION,
            AttackStage.INITIAL_RECON
        ],
        "initial_access": [
            AttackStage.OS_DETECTION,
            AttackStage.INITIAL_RECON,
            AttackStage.CREDENTIAL_HARVEST,
            AttackStage.PRIVILEGE_ESCALATION
        ],
        "post_exploit_focus": [
            AttackStage.OS_DETECTION,
            AttackStage.CREDENTIAL_HARVEST,
            AttackStage.PRIVILEGE_ESCALATION,
            AttackStage.LATERAL_MOVEMENT,
            AttackStage.DATA_COLLECTION
        ],
        "persistence_establishment": [
            AttackStage.OS_DETECTION,
            AttackStage.PERSISTENCE
        ],
        "data_exfiltration": [
            AttackStage.OS_DETECTION,
            AttackStage.DATA_COLLECTION,
            AttackStage.DATA_EXFILTRATION
        ],
        "full_engagement": None  # None means all stages
    }
    
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
        self._os_detection = None
        self._recon = None
        self._post_exploit = None
        self._exploit = None
        self._persistence = None
        self._evasion = None
        self.detected_os = None  # Store detected OS info
        self.stage_results: Dict[str, Dict] = {}
        self.chain_complete = False
    
    @property
    def os_detection(self):
        """Lazy load OS detection module"""
        if self._os_detection is None:
            from core.modules.os_detection import OSDetection
            self._os_detection = OSDetection(self.logger)
        return self._os_detection
    
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
            if stage == AttackStage.OS_DETECTION:
                # Detect OS first (local and remote)
                local_os = self.os_detection.detect_local_os()
                stage_results["results"]["local_os"] = local_os
                
                # Detect remote OS if target is specified
                if self.target:
                    remote_os = self.os_detection.detect_remote_os(self.target)
                    stage_results["results"]["remote_os"] = remote_os
                    self.detected_os = remote_os.get("os_type", "Unknown")
                    self.logger.info(f"Detected remote OS: {self.detected_os}")
                else:
                    self.detected_os = local_os.get("os_type", "Unknown")
                    self.logger.info(f"Detected local OS: {self.detected_os}")
                
            elif stage == AttackStage.INITIAL_RECON:
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
    
    def execute_full_chain(self, stages: List[AttackStage] = None, profile: Optional[str] = None, 
                          exclude_stages: Optional[List[AttackStage]] = None) -> Dict:
        """
        Execute full attack chain with optional module selection
        
        Args:
            stages: Optional list of stages to execute (defaults to all)
            profile: Optional attack profile name (recon_only, initial_access, post_exploit_focus, etc.)
            exclude_stages: Optional list of stages to exclude from execution
        
        Returns:
            Complete chain results
        
        Examples:
            # Use a predefined profile
            results = chain.execute_full_chain(profile="recon_only")
            
            # Select specific stages
            results = chain.execute_full_chain(stages=[
                AttackStage.OS_DETECTION,
                AttackStage.INITIAL_RECON,
                AttackStage.CREDENTIAL_HARVEST
            ])
            
            # Use profile but exclude cleanup
            results = chain.execute_full_chain(profile="full_engagement", exclude_stages=[AttackStage.CLEANUP])
        """
        # Use profile if specified
        if profile:
            if profile not in self.ATTACK_PROFILES:
                self.logger.warning(f"Unknown profile '{profile}', using default. Available: {list(self.ATTACK_PROFILES.keys())}")
                stages = list(AttackStage)
            else:
                profile_stages = self.ATTACK_PROFILES[profile]
                if profile_stages is None:
                    # Full engagement profile
                    stages = list(AttackStage)
                else:
                    stages = profile_stages
        
        # Default to all stages if not specified
        if stages is None:
            stages = list(AttackStage)
        
        # Remove excluded stages
        if exclude_stages:
            stages = [s for s in stages if s not in exclude_stages]
        
        self.logger.info(f"Starting attack chain execution with {len(stages)} stages")
        if profile:
            self.logger.info(f"Using attack profile: {profile}")
        
        chain_results = {
            "start_time": time.time(),
            "stages": [],
            "overall_status": "in_progress",
            "detected_os": None,
            "profile_used": profile,
            "stages_executed": [s.value for s in stages]
        }
        
        # Always start with OS detection
        if AttackStage.OS_DETECTION not in stages:
            os_result = self.execute_stage(AttackStage.OS_DETECTION)
            chain_results["stages"].append(os_result)
            chain_results["detected_os"] = self.detected_os
        
        for stage in stages:
            if stage == AttackStage.CLEANUP:
                # Always execute cleanup last
                continue
            
            if stage == AttackStage.OS_DETECTION:
                # Already executed above
                continue
            
            result = self.execute_stage(stage, delay=random.uniform(0.5, 2.0))
            chain_results["stages"].append(result)
        
        # Execute cleanup if it was in the original stages list
        if AttackStage.CLEANUP in stages:
            cleanup_result = self.execute_stage(AttackStage.CLEANUP)
            chain_results["stages"].append(cleanup_result)
        
        chain_results["end_time"] = time.time()
        chain_results["duration"] = chain_results["end_time"] - chain_results["start_time"]
        chain_results["overall_status"] = "completed"
        self.chain_complete = True
        
        self.logger.info(f"Attack chain completed in {chain_results['duration']:.2f} seconds")
        
        return chain_results
    
    def get_available_profiles(self) -> Dict[str, List[str]]:
        """
        Get available attack profiles
        
        Returns:
            Dictionary mapping profile names to stage lists
        """
        profiles = {}
        for profile_name, stages in self.ATTACK_PROFILES.items():
            if stages is None:
                profiles[profile_name] = [s.value for s in AttackStage]
            else:
                profiles[profile_name] = [s.value for s in stages]
        return profiles
    
    def create_custom_profile(self, profile_name: str, stages: List[AttackStage]) -> None:
        """
        Create a custom attack profile
        
        Args:
            profile_name: Name for the custom profile
            stages: List of stages to include in the profile
        
        Example:
            chain.create_custom_profile("custom_recon", [
                AttackStage.OS_DETECTION,
                AttackStage.INITIAL_RECON
            ])
        """
        self.ATTACK_PROFILES[profile_name] = stages
        self.logger.info(f"Created custom profile: {profile_name} with {len(stages)} stages")
    
    def execute_on_targets(self, targets: List[str], profile: Optional[str] = None, 
                          sequential: bool = True) -> Dict:
        """
        Execute attack chain on multiple targets sequentially (stealthy)
        
        Args:
            targets: List of target IP addresses
            profile: Attack profile to use (recon_only, initial_access, etc.)
            sequential: Execute sequentially (True) or parallel (False)
                       Sequential is more stealthy, default: True
        
        Returns:
            Dictionary with results for each target
        
        Example:
            targets = ["192.168.1.100", "192.168.1.101"]
            results = chain.execute_on_targets(targets, profile="recon_only", sequential=True)
        """
        if not targets:
            self.logger.warning("No targets provided")
            return {
                "targets": [],
                "results": {},
                "total_targets": 0,
                "completed": 0,
                "error": "No targets provided"
            }
        
        self.logger.info(f"Executing attack chain on {len(targets)} target(s)")
        self.logger.info(f"Execution mode: {'sequential (stealthy)' if sequential else 'parallel (fast)'}")
        
        all_results = {
            "targets": targets,
            "results": {},
            "total_targets": len(targets),
            "completed": 0,
            "failed": 0,
            "profile_used": profile,
            "sequential": sequential,
            "timestamp": time.time()
        }
        
        for i, target_ip in enumerate(targets, 1):
            self.logger.info(f"[{i}/{len(targets)}] Executing on {target_ip}")
            
            try:
                # Update target for this iteration
                original_target = self.target
                self.target = target_ip
                
                # Execute attack chain
                results = self.execute_full_chain(profile=profile)
                all_results["results"][target_ip] = results
                all_results["completed"] += 1
                
                self.logger.info(f"[{i}/{len(targets)}] Completed {target_ip}: {results['overall_status']}")
                
                # Restore original target
                self.target = original_target
                
                # Stealthy delay between targets (sequential only)
                if sequential and i < len(targets):
                    delay = random.uniform(2.0, 5.0)  # Random delays between 2-5 seconds
                    self.logger.info(f"Waiting {delay:.1f}s before next target (stealthy delay)...")
                    time.sleep(delay)
                    
            except Exception as e:
                self.logger.error(f"Failed to execute on {target_ip}: {e}")
                all_results["results"][target_ip] = {
                    "status": "failed",
                    "error": str(e)
                }
                all_results["failed"] += 1
        
        all_results["end_time"] = time.time()
        all_results["duration"] = all_results["end_time"] - all_results["timestamp"]
        
        self.logger.info(f"Multi-target execution complete: {all_results['completed']}/{all_results['total_targets']} succeeded")
        
        return all_results
    
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

