"""Run memory integration tests."""

import subprocess
import sys

def run_tests():
    """Run the memory integration tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "Bella/tests/integration/memory_integration_test.py",
        "-vvs",
        "--capture=no"
    ]
    
    print("Running memory integration tests...")
    result = subprocess.run(cmd, check=False)
    
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with code {result.returncode}")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())
