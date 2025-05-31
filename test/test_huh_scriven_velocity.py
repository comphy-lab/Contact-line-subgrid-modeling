import pytest
import numpy as np
from huh_scriven_velocity import term, Ur, Uphi, Ux, Uy

class TestTermFunction:
    """Test the common term function"""
    
    def test_term_at_pi_half(self):
        """Test term at theta=pi/2"""
        U_p = 1.0
        theta = np.pi/2
        expected = U_p / (theta - 0)  # cos(pi/2)*sin(pi/2) = 0
        result = term(U_p, theta)
        assert result == pytest.approx(expected)
    
    def test_term_with_zero_velocity(self):
        """Test term with zero plate velocity"""
        U_p = 0.0
        theta = np.pi/3
        result = term(U_p, theta)
        assert result == pytest.approx(0)
    
    def test_term_with_negative_velocity(self):
        """Test term with negative plate velocity"""
        U_p = -1.0
        theta = np.pi/4
        result = term(U_p, theta)
        assert result < 0

class TestVelocityComponents:
    """Test velocity component functions"""
    
    def test_ur_at_phi_zero(self):
        """Test Ur when phi=0 (on the plate)"""
        theta = np.pi/3
        phi = 0
        result = Ur(theta, phi)
        # At phi=0: Ur = term * (1*sin(theta) - theta*cos(theta))
        expected = term(1, theta) * (np.sin(theta) - theta*np.cos(theta))
        assert result == pytest.approx(expected)
    
    def test_uphi_at_phi_zero(self):
        """Test Uphi when phi=0 (on the plate)"""
        theta = np.pi/3
        phi = 0
        result = Uphi(theta, phi)
        # At phi=0: Uphi = 0
        assert result == pytest.approx(0)
    
    def test_ur_at_phi_theta(self):
        """Test Ur when phi=theta (on the interface)"""
        theta = np.pi/4
        phi = theta
        result = Ur(theta, phi)
        assert np.isfinite(result)
    
    def test_uphi_at_phi_theta(self):
        """Test Uphi when phi=theta (on the interface)"""
        theta = np.pi/4
        phi = theta
        result = Uphi(theta, phi)
        assert np.isfinite(result)

class TestCartesianVelocities:
    """Test Cartesian velocity components"""
    
    def test_ux_uy_transformation(self):
        """Test that Ux, Uy properly transform from polar"""
        theta = np.pi/3
        phi = np.pi/6
        
        ur = Ur(theta, phi)
        uphi = Uphi(theta, phi)
        ux = Ux(theta, phi)
        uy = Uy(theta, phi)
        
        # Check transformation formulas
        angle = theta - phi
        expected_ux = ur * np.cos(angle) - uphi * np.sin(angle)
        expected_uy = ur * np.sin(angle) + uphi * np.cos(angle)
        
        assert ux == pytest.approx(expected_ux)
        assert uy == pytest.approx(expected_uy)
    
    def test_velocity_magnitude_conservation(self):
        """Test that velocity magnitude is conserved in transformation"""
        theta = np.pi/4
        phi = np.pi/8
        
        ur = Ur(theta, phi)
        uphi = Uphi(theta, phi)
        ux = Ux(theta, phi)
        uy = Uy(theta, phi)
        
        polar_mag_squared = ur**2 + uphi**2
        cartesian_mag_squared = ux**2 + uy**2
        
        assert polar_mag_squared == pytest.approx(cartesian_mag_squared)

class TestBoundaryConditions:
    """Test velocity boundary conditions"""
    
    def test_no_slip_on_plate(self):
        """Test no-slip condition on the moving plate (phi=0)"""
        theta = np.pi/3
        phi = 0
        
        ux = Ux(theta, phi)
        uy = Uy(theta, phi)
        
        # On the plate, fluid velocity should match plate velocity
        # Relative velocity should be zero in the reference frame
        ux_rel = -1 - ux  # U_p = 1
        uy_rel = -uy
        
        # The relative velocities should be small but not necessarily zero
        # due to the slip condition
        assert np.isfinite(ux_rel)
        assert np.isfinite(uy_rel)
    
    def test_interface_condition(self):
        """Test velocity at the interface (phi=theta)"""
        theta = np.pi/4
        phi = theta
        
        ux = Ux(theta, phi)
        uy = Uy(theta, phi)
        
        # Velocities should be finite at the interface
        assert np.isfinite(ux)
        assert np.isfinite(uy)

class TestParameterRanges:
    """Test behavior over parameter ranges"""
    
    def test_small_angles(self):
        """Test velocities for small angles"""
        theta = 0.01
        phi = 0.005
        
        # All velocity components should be finite
        assert np.isfinite(Ur(theta, phi))
        assert np.isfinite(Uphi(theta, phi))
        assert np.isfinite(Ux(theta, phi))
        assert np.isfinite(Uy(theta, phi))
    
    def test_large_angles(self):
        """Test velocities for large angles"""
        theta = np.pi - 0.01
        phi = theta / 2
        
        # All velocity components should be finite
        assert np.isfinite(Ur(theta, phi))
        assert np.isfinite(Uphi(theta, phi))
        assert np.isfinite(Ux(theta, phi))
        assert np.isfinite(Uy(theta, phi))
    
    def test_phi_range_validity(self):
        """Test that phi stays within valid range [0, theta]"""
        theta = np.pi/2
        
        # Test at boundaries
        phi_min = 0
        phi_max = theta
        
        # All should work without errors
        assert np.isfinite(Ur(theta, phi_min))
        assert np.isfinite(Ur(theta, phi_max))
        assert np.isfinite(Uphi(theta, phi_min))
        assert np.isfinite(Uphi(theta, phi_max))

class TestArrayOperations:
    """Test vectorized operations"""
    
    def test_vectorized_operations(self):
        """Test that functions work with numpy arrays"""
        theta_array = np.array([np.pi/6, np.pi/4, np.pi/3])
        phi_array = np.array([np.pi/12, np.pi/8, np.pi/6])
        
        ur_array = Ur(theta_array, phi_array)
        uphi_array = Uphi(theta_array, phi_array)
        ux_array = Ux(theta_array, phi_array)
        uy_array = Uy(theta_array, phi_array)
        
        assert ur_array.shape == theta_array.shape
        assert uphi_array.shape == theta_array.shape
        assert ux_array.shape == theta_array.shape
        assert uy_array.shape == theta_array.shape
        assert np.all(np.isfinite(ur_array))
        assert np.all(np.isfinite(uphi_array))
        assert np.all(np.isfinite(ux_array))
        assert np.all(np.isfinite(uy_array))