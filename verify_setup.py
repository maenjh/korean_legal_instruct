#!/usr/bin/env python3
"""
Verification script to check project structure and configuration.
This script doesn't require external dependencies.
"""
import os
import sys
import yaml


def check_directory_structure():
    """Verify all required directories exist."""
    print("Checking directory structure...")
    
    required_dirs = [
        'src/data',
        'src/models',
        'src/training',
        'src/inference',
        'configs',
        'scripts',
        'examples'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")
            return False
    
    return True


def check_configuration_files():
    """Verify all configuration files exist and are valid."""
    print("\nChecking configuration files...")
    
    config_files = [
        'configs/base_config.yaml',
        'configs/civil_config.yaml',
        'configs/criminal_config.yaml',
        'configs/ip_config.yaml',
        'configs/administrative_config.yaml',
        'configs/router_config.yaml'
    ]
    
    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"  ✗ {config_file} (missing)")
            return False
        
        try:
            with open(config_file, 'r') as f:
                yaml.safe_load(f)
            print(f"  ✓ {config_file}")
        except Exception as e:
            print(f"  ✗ {config_file} (invalid YAML: {e})")
            return False
    
    return True


def check_source_files():
    """Verify all source files exist and compile."""
    print("\nChecking source files...")
    
    source_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/data/data_loader.py',
        'src/models/__init__.py',
        'src/models/lora_model.py',
        'src/models/router.py',
        'src/training/__init__.py',
        'src/training/lora_trainer.py',
        'src/training/router_trainer.py',
        'src/inference/__init__.py',
        'src/inference/pipeline.py'
    ]
    
    for source_file in source_files:
        if not os.path.exists(source_file):
            print(f"  ✗ {source_file} (missing)")
            return False
        
        try:
            with open(source_file, 'r') as f:
                compile(f.read(), source_file, 'exec')
            print(f"  ✓ {source_file}")
        except SyntaxError as e:
            print(f"  ✗ {source_file} (syntax error: {e})")
            return False
    
    return True


def check_scripts():
    """Verify all scripts exist and compile."""
    print("\nChecking scripts...")
    
    script_files = [
        'scripts/prepare_data.py',
        'scripts/train_lora.py',
        'scripts/train_router.py',
        'examples/mvp_demo.py',
        'examples/test_router.py'
    ]
    
    for script_file in script_files:
        if not os.path.exists(script_file):
            print(f"  ✗ {script_file} (missing)")
            return False
        
        try:
            with open(script_file, 'r') as f:
                compile(f.read(), script_file, 'exec')
            
            # Check if executable
            is_executable = os.access(script_file, os.X_OK)
            status = "✓" if is_executable else "⚠"
            print(f"  {status} {script_file}")
        except SyntaxError as e:
            print(f"  ✗ {script_file} (syntax error: {e})")
            return False
    
    return True


def check_documentation():
    """Verify documentation files exist."""
    print("\nChecking documentation...")
    
    doc_files = [
        'README.md',
        'QUICKSTART.md',
        'requirements.txt',
        '.gitignore'
    ]
    
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            print(f"  ✓ {doc_file}")
        else:
            print(f"  ✗ {doc_file} (missing)")
            return False
    
    return True


def verify_configuration_content():
    """Verify configuration content."""
    print("\nVerifying configuration content...")
    
    # Check base config
    with open('configs/base_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    required_keys = ['base_model', 'lora_config', 'training', 'data']
    for key in required_keys:
        if key in base_config:
            print(f"  ✓ base_config has '{key}'")
        else:
            print(f"  ✗ base_config missing '{key}'")
            return False
    
    # Check domain configs
    domains = ['civil', 'criminal', 'ip', 'administrative']
    for domain in domains:
        with open(f'configs/{domain}_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        if config.get('domain') == domain:
            print(f"  ✓ {domain}_config domain is correct")
        else:
            print(f"  ✗ {domain}_config domain mismatch")
            return False
    
    # Check router config
    with open('configs/router_config.yaml', 'r') as f:
        router_config = yaml.safe_load(f)
    
    if 'labels' in router_config and len(router_config['labels']) == 4:
        print(f"  ✓ router_config has 4 domain labels")
    else:
        print(f"  ✗ router_config labels incorrect")
        return False
    
    return True


def main():
    """Run all verification checks."""
    print("="*60)
    print("Korean Legal Instruct Project Verification")
    print("="*60)
    print()
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Configuration Files", check_configuration_files),
        ("Source Files", check_source_files),
        ("Scripts", check_scripts),
        ("Documentation", check_documentation),
        ("Configuration Content", verify_configuration_content)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nError during {name} check: {e}")
            results.append((name, False))
        print()
    
    # Summary
    print("="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print()
    if all_passed:
        print("✓ All checks passed! Project is ready for use.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Prepare data: python scripts/prepare_data.py")
        print("  3. Train models: see QUICKSTART.md for details")
        return 0
    else:
        print("✗ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
