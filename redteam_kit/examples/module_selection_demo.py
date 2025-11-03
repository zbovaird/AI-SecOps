#!/usr/bin/env python3
"""
Attack Chain Module Selection Examples
Demonstrates how to use attack profiles and custom module selection
"""

from utils.logger import FrameworkLogger
from core.modules.attack_chain import AttackChain, AttackStage

def main():
    logger = FrameworkLogger("module_selection_demo")
    target = "TARGET_IP_HERE"  # Replace with your target IP address
    chain = AttackChain(logger, target=target)
    
    print("=" * 60)
    print("Attack Chain Module Selection Examples")
    print("=" * 60)
    
    # Example 1: List available profiles
    print("\n1. Available Attack Profiles:")
    profiles = chain.get_available_profiles()
    for name, stages in profiles.items():
        print(f"   {name}: {len(stages)} stages")
        print(f"      Stages: {', '.join(stages[:3])}...")
    
    # Example 2: Use predefined profile
    print("\n2. Using 'recon_only' profile:")
    print("   (This would run OS detection + initial recon)")
    # Uncomment to actually run:
    # results = chain.execute_full_chain(profile="recon_only")
    # print(f"   Status: {results['overall_status']}")
    # print(f"   Stages executed: {len(results['stages'])}")
    
    # Example 3: Custom stage selection
    print("\n3. Custom stage selection:")
    custom_stages = [
        AttackStage.OS_DETECTION,
        AttackStage.INITIAL_RECON,
        AttackStage.CREDENTIAL_HARVEST
    ]
    print(f"   Selected {len(custom_stages)} stages:")
    for stage in custom_stages:
        print(f"      - {stage.value}")
    # Uncomment to actually run:
    # results = chain.execute_full_chain(stages=custom_stages)
    
    # Example 4: Create custom profile
    print("\n4. Creating custom profile:")
    chain.create_custom_profile("web_app_test", [
        AttackStage.OS_DETECTION,
        AttackStage.INITIAL_RECON,
        AttackStage.CREDENTIAL_HARVEST
    ])
    print("   Created profile: 'web_app_test'")
    
    # Example 5: Use profile with exclusions
    print("\n5. Using profile with exclusions:")
    print("   Using 'full_engagement' but excluding cleanup")
    # Uncomment to actually run:
    # results = chain.execute_full_chain(
    #     profile="full_engagement",
    #     exclude_stages=[AttackStage.CLEANUP]
    # )
    
    print("\n" + "=" * 60)
    print("Examples complete! Uncomment code to actually execute.")
    print("=" * 60)

if __name__ == "__main__":
    main()

