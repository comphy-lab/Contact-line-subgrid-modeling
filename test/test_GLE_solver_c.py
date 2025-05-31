import os
import subprocess
import numpy as np
import pytest

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_FILE = os.path.join(SRC, 'GLE_solver.c')
HEADER_DIR = os.path.join(SRC, 'src-local')


def compile_executable(tmp_path):
  exe = tmp_path / 'gle_solver_exec'
  cmd = ['gcc', C_FILE, '-Isrc-local', '-lm', '-o', str(exe)]
  subprocess.run(cmd, cwd=SRC, check=True)
  return exe


def test_compile_and_run(tmp_path):
  exe = compile_executable(tmp_path)
  csv_path = tmp_path / 'out.csv'
  subprocess.run([str(exe), str(csv_path)], check=True)
  assert csv_path.exists()
  data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
  assert data.shape[1] == 3
  s = data[:, 0]
  h = data[:, 1]
  theta = data[:, 2]
  assert np.isclose(s[0], 0.0)
  assert h[0] > 0
  assert np.all(np.isfinite(theta))
  assert np.all(np.diff(s) > 0)
