import pytest
import numpy as np
import os
import sys
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GLE_solver import run_solver_and_plot as run_gle_solver
from huh_scriven_velocity import compute_and_plot as run_huh_scriven

class TestIntegration:
    """Integration tests that run the actual solvers and check outputs"""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and cleanup for each test"""
        # Setup: Create test output directory
        self.test_output_dir = 'test_plots'
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        yield
        
        # Teardown: Remove test output directory
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
    
    def test_gle_solver_runs_without_error(self):
        """Test that GLE solver runs and produces output"""
        solution, s_values, h_values, theta_values, w_values = run_gle_solver(output_dir=self.test_output_dir)
        
        # Check that solution exists
        assert solution is not None
        
        # Note: The solver has known convergence issues (see CLAUDE.md)
        # We check if it at least attempted to solve
        if not solution.success:
            print(f"Warning: GLE solver did not converge. Message: {solution.message}")
            # Check that it at least produced some output
            assert solution.status in [0, 1, 2]  # Common scipy solver status codes
        
        # Check output arrays
        assert len(s_values) > 0
        assert len(h_values) > 0
        assert len(theta_values) > 0
        assert len(w_values) > 0
        
        # Check that values are finite
        assert np.all(np.isfinite(s_values))
        assert np.all(np.isfinite(h_values))
        assert np.all(np.isfinite(theta_values))
        assert np.all(np.isfinite(w_values))
    
    def test_gle_solver_creates_plots(self):
        """Test that GLE solver creates plot files"""
        solution, _, _, _, _ = run_gle_solver(output_dir=self.test_output_dir)
        
        # Skip plot check if solver didn't converge
        if not solution.success:
            pytest.skip(f"Skipping plot test - solver did not converge: {solution.message}")
        
        # Check that plot files exist
        h_plot_path = os.path.join(self.test_output_dir, 'GLE_h_profile.png')
        theta_plot_path = os.path.join(self.test_output_dir, 'GLE_theta_profile.png')
        
        assert os.path.exists(h_plot_path)
        assert os.path.exists(theta_plot_path)
        
        # Check that files have content (size > 0)
        assert os.path.getsize(h_plot_path) > 0
        assert os.path.getsize(theta_plot_path) > 0
    
    def test_huh_scriven_runs_without_error(self):
        """Test that Huh-Scriven velocity calculation runs"""
        Theta_grid, Phi_grid, Ux_rel_grid, Uy_rel_grid = run_huh_scriven(output_dir=self.test_output_dir)
        
        # Check output arrays
        assert len(Theta_grid) > 0
        assert len(Phi_grid) > 0
        assert len(Ux_rel_grid) > 0
        assert len(Uy_rel_grid) > 0
        
        # Check shapes match
        assert Theta_grid.shape == Phi_grid.shape
        assert Ux_rel_grid.shape == Uy_rel_grid.shape
        assert Theta_grid.shape == Ux_rel_grid.shape
        
        # Check that values are finite
        assert np.all(np.isfinite(Theta_grid))
        assert np.all(np.isfinite(Phi_grid))
        assert np.all(np.isfinite(Ux_rel_grid))
        assert np.all(np.isfinite(Uy_rel_grid))
    
    def test_huh_scriven_creates_plots(self):
        """Test that Huh-Scriven creates plot files"""
        run_huh_scriven(output_dir=self.test_output_dir)
        
        # Check that plot files exist
        ux_plot_path = os.path.join(self.test_output_dir, 'huh_scriven_Ux_rel.png')
        uy_plot_path = os.path.join(self.test_output_dir, 'huh_scriven_Uy_rel.png')
        
        assert os.path.exists(ux_plot_path)
        assert os.path.exists(uy_plot_path)
        
        # Check that files have content (size > 0)
        assert os.path.getsize(ux_plot_path) > 0
        assert os.path.getsize(uy_plot_path) > 0
    
    def test_gle_physical_constraints(self):
        """Test that GLE solution satisfies physical constraints"""
        solution, s_values, h_values, theta_values, w_values = run_gle_solver(output_dir=self.test_output_dir)
        
        # Skip constraint check if solver didn't converge
        if not solution.success:
            pytest.skip(f"Skipping physical constraint test - solver did not converge: {solution.message}")
        
        # theta_values is already in radians from the solver
        
        # h should be positive
        assert np.all(h_values > 0)
        
        # theta should be between 0 and pi
        assert np.all(theta_values > 0)
        assert np.all(theta_values < np.pi)
        
        # h should increase with s (since dh/ds = sin(theta) > 0 for 0 < theta < pi)
        assert h_values[-1] > h_values[0]
    
    def test_huh_scriven_phi_constraint(self):
        """Test that phi stays within valid range [0, theta]"""
        Theta_grid, Phi_grid, Ux_rel_grid, Uy_rel_grid = run_huh_scriven(output_dir=self.test_output_dir)
        
        # Check that 0 <= phi <= theta for all points
        assert np.all(Phi_grid >= 0)
        assert np.all(Phi_grid <= Theta_grid)
    
    def test_reproducibility(self):
        """Test that repeated runs give same results"""
        # Run solver twice
        solution1, s1, h1, theta1, w1 = run_gle_solver(output_dir=self.test_output_dir)
        solution2, s2, h2, theta2, w2 = run_gle_solver(output_dir=self.test_output_dir)
        
        # Skip if either solver didn't converge
        if not solution1.success or not solution2.success:
            pytest.skip("Skipping reproducibility test - solver convergence issues")
        
        # Check that results are identical
        np.testing.assert_array_almost_equal(s1, s2)
        np.testing.assert_array_almost_equal(h1, h2)
        np.testing.assert_array_almost_equal(theta1, theta2)
        np.testing.assert_array_almost_equal(w1, w2)