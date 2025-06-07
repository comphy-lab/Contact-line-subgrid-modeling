import pytest
import numpy as np
from GLE_solver import f1, f2, f3, f, GLE, boundary_conditions

class TestHelperFunctions:
    """Test the helper functions f1, f2, f3"""
    
    def test_f1_at_zero(self):
        """Test f1 at theta=0"""
        assert f1(0) == pytest.approx(0)
    
    def test_f1_at_pi(self):
        """Test f1 at theta=pi"""
        assert f1(np.pi) == pytest.approx(np.pi**2)
    
    def test_f2_at_zero(self):
        """Test f2 at theta=0"""
        assert f2(0) == pytest.approx(0)
    
    def test_f2_at_pi(self):
        """Test f2 at theta=pi"""
        assert f2(np.pi) == pytest.approx(np.pi)
    
    def test_f3_at_zero(self):
        """Test f3 at theta=0"""
        assert f3(0) == pytest.approx(0)
    
    def test_f3_at_pi(self):
        """Test f3 at theta=pi"""
        assert f3(np.pi) == pytest.approx(0)
    
    def test_f3_at_pi_half(self):
        """Test f3 at theta=pi/2"""
        theta = np.pi/2
        expected = theta * (np.pi - theta) + 1
        assert f3(theta) == pytest.approx(expected)

class TestFFunction:
    """Test the combined f(theta, mu_r) function"""
    
    def test_f_near_pi_half(self):
        """Test f near theta=pi/2 (avoiding singularity)"""
        theta = np.pi/2 - 0.01  # Slightly off pi/2 to avoid division by zero
        mu_r = 1.0
        result = f(theta, mu_r)
        assert isinstance(result, float)
        assert np.isfinite(result)
    
    def test_f_with_small_mu_r(self):
        """Test f with small viscosity ratio"""
        theta = np.pi/3
        mu_r = 1e-3
        result = f(theta, mu_r)
        assert np.isfinite(result)
    
    def test_f_with_large_mu_r(self):
        """Test f with large viscosity ratio"""
        theta = np.pi/4
        mu_r = 1000
        result = f(theta, mu_r)
        assert np.isfinite(result)

class TestGLESystem:
    """Test the GLE ODE system"""
    
    def test_GLE_output_shape(self):
        """Test that GLE returns correct shape"""
        s = 0.5
        y = [1e-5, np.pi/6, 0.1]  # h, theta, omega
        Ca = 0.01
        mu_r = 1e-6
        lambda_slip = 1e-4
        result = GLE(s, y, Ca, mu_r, lambda_slip)
        assert len(result) == 3
    
    def test_GLE_first_equation(self):
        """Test dh/ds = sin(theta)"""
        s = 0.5
        h = 1e-5
        theta = np.pi/4
        omega = 0.1
        y = [h, theta, omega]
        Ca = 0.01
        mu_r = 1e-6
        lambda_slip = 1e-4
        result = GLE(s, y, Ca, mu_r, lambda_slip)
        assert result[0] == pytest.approx(np.sin(theta))
    
    def test_GLE_second_equation(self):
        """Test dtheta/ds = omega"""
        s = 0.5
        h = 1e-5
        theta = np.pi/4
        omega = 0.15
        y = [h, theta, omega]
        Ca = 0.01
        mu_r = 1e-6
        lambda_slip = 1e-4
        result = GLE(s, y, Ca, mu_r, lambda_slip)
        assert result[1] == pytest.approx(omega)
    
    def test_GLE_with_zero_curvature(self):
        """Test GLE with zero initial curvature"""
        s = 0.5
        h = 1e-5
        theta = np.pi/6
        omega = 0
        y = [h, theta, omega]
        Ca = 0.01
        mu_r = 1e-6
        lambda_slip = 1e-4
        result = GLE(s, y, Ca, mu_r, lambda_slip)
        assert len(result) == 3
        # Check that omega derivative is non-zero (due to gravity term)
        assert result[2] != 0
        assert np.all(np.isfinite(result))

class TestBoundaryConditions:
    """Test the boundary condition function"""
    
    def test_boundary_conditions_shape(self):
        """Test that boundary_conditions returns correct shape"""
        ya = [1e-5, np.pi/6, 0]  # Values at s=0
        yb = [1e-4, np.pi/4, 0]  # Values at s=Delta
        w_bc = 0  # Test boundary condition value
        theta0 = np.pi/6
        lambda_slip = 1e-5
        result = boundary_conditions(ya, yb, w_bc, theta0, lambda_slip)
        assert len(result) == 3
    
    def test_boundary_conditions_satisfied(self):
        """Test that correct boundary values satisfy conditions"""
        theta0 = np.pi/6
        lambda_slip = 1e-5
        w_bc = 0
        ya = [lambda_slip, theta0, 0.5]  # Correct values at s=0
        yb = [1e-4, np.pi/4, w_bc]  # Correct omega at s=Delta
        result = boundary_conditions(ya, yb, w_bc, theta0, lambda_slip)
        assert result[0] == pytest.approx(0)  # theta condition
        assert result[1] == pytest.approx(0)  # h condition
        assert result[2] == pytest.approx(0)  # omega condition

class TestParameterRanges:
    """Test behavior with different parameter ranges"""
    
    def test_extreme_capillary_number(self):
        """Test with very small and large Ca"""
        theta = np.pi/4
        
        # The function should handle extreme values
        result_small = f(theta, 1e-10)
        result_large = f(theta, 1e10)
        
        assert np.isfinite(result_small)
        assert np.isfinite(result_large)
    
    def test_theta_range(self):
        """Test f1, f2, f3 over valid theta range"""
        theta_values = np.linspace(0.01, np.pi - 0.01, 100)
        
        for theta in theta_values:
            assert np.isfinite(f1(theta))
            assert np.isfinite(f2(theta))
            assert np.isfinite(f3(theta))
    
    def test_numerical_stability_near_boundaries(self):
        """Test numerical stability near theta=0 and theta=pi"""
        mu_r = 0.1
        
        # Near zero
        theta_small = 1e-6
        result_small = f(theta_small, mu_r)
        assert np.isfinite(result_small)
        
        # Near pi
        theta_large = np.pi - 1e-6
        result_large = f(theta_large, mu_r)
        assert np.isfinite(result_large)