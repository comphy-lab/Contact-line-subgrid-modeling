"""
Quick test script for GLE_continuation_v4.py
Shows that both methods work correctly.
"""

import subprocess
import sys
import os

def run_continuation(method, mu_r, lambda_slip, theta0, max_steps=30):
  """Run continuation with given parameters."""
  cmd = [
    sys.executable, 'GLE_continuation_v4.py',
    '--method', method,
    '--mu_r', str(mu_r),
    '--lambda_slip', str(lambda_slip),
    '--theta0', str(theta0),
    '--max-steps', str(max_steps),
    '--output-dir', f'output/test_{method}'
  ]
  
  print(f"\nRunning {method} continuation:")
  print(f"  Command: {' '.join(cmd)}")
  
  result = subprocess.run(cmd, capture_output=True, text=True)
  
  if result.returncode == 0:
    print("  SUCCESS")
    # Extract summary from output
    lines = result.stdout.strip().split('\n')
    for line in lines[-10:]:
      if 'Critical Ca' in line or 'Estimated Ca_critical' in line:
        print(f"  {line}")
  else:
    print(f"  FAILED: {result.stderr}")
    
  return result.returncode == 0


def main():
  """Run tests for both continuation methods."""
  print("Testing GLE_continuation_v4.py")
  print("="*60)
  
  # Test 1: Natural parameter method
  success1 = run_continuation('natural', 1e-6, 1e-4, 60, max_steps=50)
  
  # Test 2: Pseudo-arclength method  
  success2 = run_continuation('arclength', 1e-6, 1e-4, 60, max_steps=50)
  
  # Test 3: Natural method with different parameters
  success3 = run_continuation('natural', 1e-3, 1e-3, 45, max_steps=30)
  
  print("\n" + "="*60)
  print("Summary:")
  print(f"  Natural method test 1: {'PASSED' if success1 else 'FAILED'}")
  print(f"  Arclength method test: {'PASSED' if success2 else 'FAILED'}")
  print(f"  Natural method test 2: {'PASSED' if success3 else 'FAILED'}")
  
  # Check output files
  print("\nOutput files created:")
  for method in ['natural', 'arclength']:
    test_dir = f'output/test_{method}'
    if os.path.exists(test_dir):
      files = os.listdir(test_dir)
      print(f"  {test_dir}/: {len(files)} files")
      for f in sorted(files):
        print(f"    - {f}")
        
  print("\nAll tests completed!")


if __name__ == '__main__':
  main()