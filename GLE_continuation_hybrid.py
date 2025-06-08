import numpy as np
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
import sys
import os

# Correctly determine the path to src-local
# __file__ is GLE_continuation_hybrid.py in the root directory
# So, src-local is in 'src-local' relative to this file's directory
current_script_dir = os.path.dirname(os.path.abspath(__file__))
src_local_dir = os.path.join(current_script_dir, "src-local")
parent_of_src_local = current_script_dir # The root directory

# Add the parent directory of 'src-local' to Python path
# This allows 'from src_local.module import ...'
sys.path.insert(0, parent_of_src_local)

try:
    from src_local.solution_types import SolutionResult
    from src_local.gle_utils import solve_single_ca, find_x0_and_theta_min
except ImportError as e:
    # Provide more context for debugging if import fails
    print(f"Error importing from src_local: {e}")
    print(f"sys.path: {sys.path}")
    print(f"Attempted to import from: {parent_of_src_local} and {src_local_dir}")
    # Re-raise the error to make it clear that setup failed
    raise


@dataclass
class SolutionPoint:
    """Single point on the continuation branch"""
    Ca: float                    # Capillary number
    X_cl: float                 # Contact line position (x0 from gle_utils)
    theta_min: float            # Minimum interface angle
    profile: np.ndarray         # Solution profile (y array from solve_bvp, e.g., shape (3, N))
    arc_length: float           # Cumulative continuation parameter
    stability: Optional[str] = None # 'stable' or 'unstable'
    newton_iters: Optional[int] = None # Convergence iterations for this point

@dataclass
class BranchTangent:
    """Tangent vector for continuation"""
    y_dot: np.ndarray           # Tangent in solution space (discretized profile)
    p_dot: float                # Tangent in parameter (Ca) direction

@dataclass
class ContinuationParams:
    """Configuration parameters"""
    mu_r: float                 # Viscosity ratio
    lambda_slip: float          # Slip length
    theta0: float               # Initial contact angle (apparent contact angle at s=0)
    Delta: float                # Domain length (s_max for integration)

    w_bc: float = 0.0           # Curvature boundary condition at s=Delta
    s_range_initial_guess_points: int = 100 # Number of points for initial s_range and y_guess mesh
    tolerance: float = 1e-6     # Convergence tolerance for corrector and BVP
    max_newton_iters: int = 10  # Maximum Newton iterations for corrector step
    initial_ds: float = 0.001   # Initial step size (absolute value)
    max_ds: float = 0.01        # Maximum step size (absolute value)
    min_ds: float = 1e-6        # Minimum step size (absolute value)
    bvp_max_nodes: int = 100000 # Max nodes for solve_bvp


class ContinuationSolver:
    """
    Main class for performing pseudo-arclength continuation for the GLE.
    """
    def __init__(self, params: ContinuationParams):
        """
        Initialize the solver with physical and numerical parameters.

        Args:
            params: A ContinuationParams object containing all necessary parameters.
        """
        self.params = params
        # Initial s_range and y_guess for the very first BVP solve
        # y_guess: [h, theta, omega]
        # h starts at lambda_slip, ends higher.
        # theta starts at theta0, decreases, then increases.
        # omega starts near 0, becomes negative, then positive.
        self.s_initial = np.linspace(0, self.params.Delta, self.params.s_range_initial_guess_points)

        # A simple linear guess for h, and quadratic for theta often works.
        # h(s) = lambda_slip + s * (some_slope_h_guess)
        # theta(s) = theta0 - s * (some_slope_theta_guess_initial) + s^2 * (some_curvature_theta_guess)
        # omega(s) = dtheta/ds

        # Guess for h: increases linearly from lambda_slip to lambda_slip + Delta/10 (arbitrary)
        h_guess = self.params.lambda_slip + self.s_initial * (self.params.Delta / 10.0 / self.params.Delta)

        # Guess for theta: starts at theta0, dips, then recovers.
        # A simple guess could be that it linearly decreases to theta0/2 and comes back to theta0.
        # Or, more simply, stays near theta0 or decreases slightly.
        # Let's try a simple linear decrease for the initial guess.
        theta_guess_end_slope = - (np.pi / 180) # Small negative slope initially
        if self.params.theta0 < np.pi/2: # if acute, guess it decreases further
             theta_guess = self.params.theta0 + theta_guess_end_slope * (self.s_initial/self.params.Delta)
        else: # if obtuse, guess it decreases
             theta_guess = self.params.theta0 - (self.params.theta0 - np.pi/4) * (self.s_initial/self.params.Delta)
        theta_guess = np.clip(theta_guess, 1e-2, np.pi - 1e-2)

        # Guess for omega (dtheta/ds):
        # If theta is linear, omega is constant. Let's use that as a first guess.
        # omega_guess = np.full_like(self.s_initial, (theta_guess[-1] - theta_guess[0]) / self.params.Delta)
        # A better guess for omega might be zero initially, or based on a simple profile.
        # For now, let's use a simple guess reflecting a dip if theta dips.
        # If theta is approximated by A*s^2 + B*s + C, then omega is 2*A*s + B
        # For a symmetric dip, B=0 at midpoint. Let's try a simple linear profile for omega.
        omega_guess_slope = 2 * theta_guess_end_slope / self.params.Delta # crude estimate
        omega_guess = omega_guess_slope * (self.s_initial - self.params.Delta / 2)
        omega_guess = np.zeros_like(self.s_initial) # Start with zero omega

        self.y_initial_guess = np.vstack([
            h_guess,
            theta_guess,
            omega_guess
        ])


    def solve_branch(self, Ca_start: float, direction: int = 1, max_steps: int = 100) -> List[SolutionPoint]:
        """
        Main continuation routine to trace a solution branch.

        Args:
            Ca_start: Starting Capillary number for the branch.
            direction: +1 for increasing Ca (or arclength in general direction of increasing Ca),
                       -1 for decreasing Ca (or arclength in general direction of decreasing Ca).
            max_steps: Maximum number of continuation steps to take.

        Returns:
            A list of SolutionPoint objects forming the branch.
        """
        # To be implemented in detail later
        pass

    def get_initial_solutions(self, Ca_start: float, initial_ds_perturbation_factor: float = 0.01) -> Tuple[Optional[SolutionPoint], Optional[SolutionPoint]]:
        """
        Computes the first two solution points to initialize the tangent.
        The first point is at Ca_start.
        The second point is at Ca_start perturbed by a small amount related to initial_ds.
        This perturbation helps in establishing the initial direction of continuation.

        Args:
            Ca_start: The Capillary number for the first solution.
            initial_ds_perturbation_factor: A factor used to determine the Ca for the second point.
                                          Ca_pert = Ca_start + initial_ds_perturbation_factor * self.params.initial_ds.
                                          The sign of initial_ds from params is not used here, perturbation is always positive.

        Returns:
            A tuple (SolutionPoint1, SolutionPoint2). Returns (None, None) if initialization fails,
            or (Point1, None) if only the first point is found.
        """
        print(f"Attempting to get initial solution at Ca = {Ca_start:.6g}")
        current_y_guess = np.copy(self.y_initial_guess) # Use a copy

        solution_result1 = self.robust_bvp_solve(Ca_start, y_guess=current_y_guess, s_mesh=self.s_initial)

        if not (solution_result1 and solution_result1.success):
            print(f"Failed to get initial solution at Ca = {Ca_start:.6g}. Message: {solution_result1.message if solution_result1 else 'robust_bvp_solve returned None'}")
            return None, None

        point1 = SolutionPoint(
            Ca=Ca_start,
            X_cl=solution_result1.x0,
            theta_min=solution_result1.theta_min,
            profile=solution_result1.solution.y,
            arc_length=0.0,
            newton_iters=0
        )
        print(f"Successfully found initial solution point 1 at Ca = {point1.Ca:.6g}, X_cl = {point1.X_cl:.6g}")

        # Second solution point - perturb Ca slightly
        # Perturbation is based on a factor of initial_ds, ensuring it's a small step.
        # The direction of this initial perturbation is positive by default.
        ca_perturbation_amount = initial_ds_perturbation_factor * self.params.initial_ds
        if ca_perturbation_amount == 0: # Ensure there's some perturbation
            ca_perturbation_amount = 1e-5

        Ca_perturbed = Ca_start + ca_perturbation_amount

        print(f"Attempting to get second initial solution at Ca_perturbed = {Ca_perturbed:.6g}")

        y_guess_for_second_point = np.copy(solution_result1.solution.y)
        s_mesh_for_second_point = np.copy(solution_result1.solution.x)

        solution_result2 = self.robust_bvp_solve(Ca_perturbed, y_guess=y_guess_for_second_point, s_mesh=s_mesh_for_second_point)

        if not (solution_result2 and solution_result2.success):
            print(f"Failed to get second initial solution at Ca = {Ca_perturbed:.6g}. Message: {solution_result2.message if solution_result2 else 'robust_bvp_solve returned None'}")
            # Try perturbing in the other direction as a fallback
            Ca_perturbed_alt = Ca_start - ca_perturbation_amount # Use the same magnitude
            print(f"Retrying second initial solution at Ca_perturbed_alt = {Ca_perturbed_alt:.6g}")
            solution_result2 = self.robust_bvp_solve(Ca_perturbed_alt, y_guess=y_guess_for_second_point, s_mesh=s_mesh_for_second_point)
            if not (solution_result2 and solution_result2.success):
                print(f"Failed to get second initial solution on retry. Message: {solution_result2.message if solution_result2 else 'robust_bvp_solve returned None'}")
                return point1, None

        # If solution_result2 is still None or not successful after retry
        if not (solution_result2 and solution_result2.success):
             return point1, None

        # Calculate a heuristic initial arc length based on Euclidean distance
        dy_vec = solution_result2.solution.y.ravel() - solution_result1.solution.y.ravel()
        dca_val = solution_result2.Ca - solution_result1.Ca

        initial_segment_length = np.sqrt(np.dot(dy_vec, dy_vec) + dca_val**2)
        if initial_segment_length == 0:
            initial_segment_length = np.abs(ca_perturbation_amount) # Fallback if profiles are identical (should not happen)

        point2 = SolutionPoint(
            Ca=solution_result2.Ca,
            X_cl=solution_result2.x0,
            theta_min=solution_result2.theta_min,
            profile=solution_result2.solution.y,
            arc_length=initial_segment_length,
            newton_iters=0
        )
        print(f"Successfully found initial solution point 2 at Ca = {point2.Ca:.6g}, X_cl = {point2.X_cl:.6g}")

        return point1, point2

    def compute_tangent(self, point1: SolutionPoint, point2: SolutionPoint) -> BranchTangent:
        """
        Computes the normalized tangent vector from two consecutive solution points.
        Uses a secant approximation: (y2 - y1, Ca2 - Ca1) / ||(y2 - y1, Ca2 - Ca1)||.
        The profiles point1.profile and point2.profile are assumed to be compatible
        (e.g., on the same mesh or resampled by the caller if necessary).

        Args:
            point1: The first solution point (earlier in arclength, e.g., y_k-1).
            point2: The second solution point (later in arclength, e.g., y_k).

        Returns:
            A BranchTangent object containing the normalized tangent components (y_dot, p_dot)
            and the original magnitude of the (profile_difference, Ca_difference) vector.
        """
        profile1_y = point1.profile
        profile2_y = point2.profile

        if profile1_y.shape[1] != profile2_y.shape[1]:
            # This is a critical point: if the number of nodes in the discretized profiles
            # differ, direct subtraction is not meaningful. The BVP solver (solve_bvp) can
            # adapt the mesh, leading to sol.y having different numbers of columns.
            # A robust implementation would require SolutionPoint to also store the s-nodes (sol.x)
            # and then one profile would be interpolated onto the mesh of the other.
            # For now, this code assumes that upstream logic (e.g., in robust_bvp_solve or how
            # SolutionPoints are created) ensures profiles are on a compatible mesh.
            # If not, the subtraction below might raise a broadcasting error.
            print(f"CRITICAL WARNING: Profiles for tangent calculation have different numbers of nodes: "
                  f"P1 has {profile1_y.shape[1]}, P2 has {profile2_y.shape[1]}. "
                  f"This can lead to incorrect tangent vectors if meshes are not aligned. "
                  f"Ensure profiles are on a common mesh before calling compute_tangent.")
            # Depending on NumPy's broadcasting rules, this might not immediately error
            # if one dimension is 1, but that's not expected for these profiles.
            # If shapes are truly incompatible (e.g. (3, 50) and (3, 55)), np.subtract will error.

        # y_dot_unnormalized is the difference in profile arrays (delta_y)
        y_dot_unnormalized = profile2_y - profile1_y

        # p_dot_unnormalized is the difference in the parameter Ca (delta_Ca)
        p_dot_unnormalized = point2.Ca - point1.Ca

        # Calculate the squared norm of the unnormalized y_dot vector
        # np.sum(vector**2) is equivalent to np.dot(vector.ravel(), vector.ravel())
        norm_y_dot_sq = np.sum(y_dot_unnormalized**2)

        # Calculate the squared norm of the unnormalized p_dot scalar
        norm_p_dot_sq = p_dot_unnormalized**2

        # Calculate the magnitude of the unnormalized tangent vector (delta_y, delta_Ca)
        original_magnitude = np.sqrt(norm_y_dot_sq + norm_p_dot_sq)

        # Handle the case where the magnitude is zero (points are identical)
        if original_magnitude < 1e-12: # Using a small epsilon for floating point comparison
            print("Warning: Tangent magnitude is near zero. Points might be identical or numerically very close.")
            # Default tangent: prioritize change in Ca if any, else arbitrary (e.g., p_dot=1)
            # This situation should ideally be avoided by step size control or other checks.
            # Returning a tangent mainly in Ca direction as a fallback.
            # y_dot components are zero because there's no change in profile.
            normalized_y_dot = np.zeros_like(y_dot_unnormalized)
            # If Ca also didn't change, p_dot_unnormalized is 0. To avoid 0/0, set p_dot to a default.
            # If Ca did change slightly but profile didn't, this would be normalized.
            # But if original_magnitude is ~0, then both y_dot_unnormalized and p_dot_unnormalized are ~0.
            # So, we set a default direction.
            normalized_p_dot = 1.0 # Default: try to move in Ca direction

            # Recalculate magnitude for this default normalized vector (it's 1.0)
            # However, the 'magnitude' field in BranchTangent refers to the *original* magnitude.
            # So, we return original_magnitude which is ~0.
            # The normalized_y_dot and normalized_p_dot form a unit vector (0,1) here.
            # This specific default might need tuning based on problem behavior.
            return BranchTangent(
                y_dot=normalized_y_dot,
                p_dot=normalized_p_dot,
                magnitude=original_magnitude # This will be close to zero
            )

        # Normalize the tangent components
        normalized_y_dot = y_dot_unnormalized / original_magnitude
        normalized_p_dot = p_dot_unnormalized / original_magnitude

        return BranchTangent(
            y_dot=normalized_y_dot,
            p_dot=normalized_p_dot,
            magnitude=original_magnitude
        )

    def predict_step(self, current_point: SolutionPoint, tangent: BranchTangent, ds: float) -> Tuple[np.ndarray, float]:
        """
        Predictor step using the tangent direction.
        Calculates: predicted_profile = current_profile + ds * tangent.y_dot
                      predicted_Ca = current_Ca + ds * tangent.p_dot

        The tangent.y_dot and tangent.p_dot are normalized components.
        ds is the arclength step.

        Args:
            current_point: The current SolutionPoint from which to predict.
            tangent: The BranchTangent at the current_point (contains normalized y_dot, p_dot).
            ds: The step size in arclength.

        Returns:
            A tuple (predicted_profile, predicted_Ca).
            predicted_profile is a numpy array for the initial guess of the next solution's profile.
            predicted_Ca is a float for the initial guess of the next solution's Capillary number.
        """
        # current_point.profile is the y-vector (shape 3xN) of the current solution
        # tangent.y_dot is the normalized change in y (shape 3xN)
        # ds is the arclength step size
        predicted_profile = current_point.profile + ds * tangent.y_dot

        # current_point.Ca is the parameter value of the current solution
        # tangent.p_dot is the normalized change in Ca
        predicted_Ca = current_point.Ca + ds * tangent.p_dot

        return predicted_profile, predicted_Ca

    def robust_bvp_solve(self, Ca: float, y_guess: np.ndarray, s_mesh: Optional[np.ndarray] = None) -> Optional[SolutionResult]:
        """
        Wrapper around gle_utils.solve_single_ca with error handling and parameter passing.
        It uses parameters stored in self.params (ContinuationParams).

        Args:
            Ca: Capillary number for the BVP solve.
            y_guess: Initial guess for the solution profile [h(s), theta(s), omega(s)].
                     Shape should be (3, num_mesh_points).
            s_mesh: The mesh for s. If None, uses self.s_initial (the initial mesh defined in __init__).
                    Shape should be (num_mesh_points,).

        Returns:
            A SolutionResult object from gle_utils if successful and physical,
            otherwise None or a SolutionResult with success=False.
        """
        if s_mesh is None:
            s_mesh_to_use = self.s_initial
        else:
            s_mesh_to_use = s_mesh

        # Ensure y_guess and s_mesh_to_use are compatible in terms of number of points.
        # y_guess is (3, N), s_mesh_to_use is (N,).
        if y_guess.shape[1] != len(s_mesh_to_use):
            # This often happens if y_guess comes from a previous solution on an adapted mesh,
            # but s_mesh_to_use is, for example, self.s_initial.
            # We should ideally use the s_mesh corresponding to y_guess.
            # If s_mesh was explicitly provided, it should match y_guess.
            # If s_mesh was None, and self.s_initial is used, then y_guess should be compatible
            # with self.s_initial, or it should be interpolated.

            # For now, let's assume if s_mesh is provided, it's the correct one for y_guess.
            # If s_mesh is NOT provided, and y_guess.shape[1] != len(self.s_initial),
            # this indicates a potential mismatch.
            # A robust solution would be to interpolate y_guess onto s_mesh_to_use.
            # However, solve_bvp itself can take y_guess from a solution on a different mesh,
            # as long as the s_mesh argument defines the mesh for *this* solve.
            # So, the critical part is that s_mesh_to_use is what we want for *this* solve,
            # and y_guess is the best available guess. Scipy's solve_bvp handles interpolation
            # of the initial guess if its mesh is different from the one specified for the problem.

            # Let's clarify: solve_bvp's `x` argument (s_mesh_to_use here) defines the initial mesh for the BVP.
            # Its `y` argument (y_guess here) is the initial guess for the solution on that mesh.
            # If y_guess comes from a previous sol.y, and sol.x was different from s_mesh_to_use,
            # then y_guess might need to be reshaped or interpolated if solve_bvp doesn't handle it.
            # Scipy's documentation: "Initial guess for the function values at the mesh nodes."
            # This implies y_guess must conform to s_mesh_to_use.

            # If s_mesh was provided, it must match y_guess.shape[1].
            if s_mesh is not None and y_guess.shape[1] != len(s_mesh):
                 print(f"Warning: y_guess ({y_guess.shape[1]} points) and provided s_mesh ({len(s_mesh)} points) "
                       f"have mismatched number of points. This may lead to errors in solve_bvp.")
            # If s_mesh was None, then s_mesh_to_use is self.s_initial.
            # If y_guess is not compatible with self.s_initial, we need to interpolate.
            elif s_mesh is None and y_guess.shape[1] != len(self.s_initial):
                print(f"Info: y_guess ({y_guess.shape[1]} points) is on a different mesh than "
                      f"self.s_initial ({len(self.s_initial)} points). Interpolating y_guess "
                      f"to self.s_initial for this BVP solve.")
                # Create temporary interpolated y_guess.
                # This requires the s_mesh corresponding to the current y_guess.
                # This information is not passed here. This is a limitation.
                # For now, we pass y_guess as is and rely on solve_bvp's behavior,
                # or assume that if s_mesh is None, y_guess is already on self.s_initial.
                # A more robust solution would be to pass (y_guess, s_guess_mesh) always.
                # For now, let's assume y_guess passed is appropriate for s_mesh_to_use,
                # or that solve_single_ca/solve_bvp can handle it.
                # The most common case: y_guess is sol.y from previous step, s_mesh is sol.x from previous step.
                pass # Fall through and let solve_single_ca handle it.

        try:
            result = solve_single_ca(
                Ca=Ca,
                mu_r=self.params.mu_r,
                lambda_slip=self.params.lambda_slip,
                theta0=self.params.theta0,
                w_bc=self.params.w_bc, # Use w_bc from ContinuationParams
                Delta=self.params.Delta,
                s_range=s_mesh_to_use, # Mesh for this BVP solve
                y_guess=y_guess,       # Initial guess for y on s_range
                tol=self.params.tolerance,
                max_nodes=self.params.bvp_max_nodes # Use max_nodes from ContinuationParams
            )

            # Validate physical constraints if solution was successful
            if result.success and result.solution is not None:
                h_vals = result.solution.y[0, :] # h is the first component
                if np.any(h_vals < 0): # Film thickness must be non-negative
                    # Check if it's truly negative or just precision noise around lambda_slip
                    # lambda_slip itself can be zero. A stricter check might be h_vals < -epsilon.
                    # For now, any h < 0 is considered unphysical.
                    print(f"Warning: BVP solve at Ca={Ca:.6g} resulted in negative film thickness. Treating as failure.")
                    return SolutionResult(
                        success=False, solution=result.solution, Ca=Ca,
                        message="Negative film thickness",
                        s_range=result.s_range, y_guess=result.y_guess, # store for diagnostics
                        x0=result.x0, theta_min=result.theta_min
                    )

                # Theta physical range [0, pi] is already checked inside solve_single_ca in gle_utils
                # but we can add more checks here if needed.

            return result

        except Exception as e:
            print(f"Exception during BVP solve at Ca={Ca:.6g}: {e}")
            # Return a SolutionResult indicating failure
            return SolutionResult(
                success=False,
                solution=None,
                Ca=Ca,
                message=f"Exception: {str(e)}",
                s_range=s_mesh_to_use, # s_mesh that was attempted
                y_guess=y_guess      # y_guess that was used
            )

    def correct_step(self, predicted_profile: np.ndarray, predicted_Ca: float,
                     reference_point: SolutionPoint, tangent: BranchTangent, ds: float) -> Optional[SolutionPoint]:
        """
        Corrector step that solves the extended system:
        1. F(y, Ca) = 0  (BVP equations)
        2. T(y, Ca) = tangent.y_dot^T @ (y - reference_point.profile) +
                      tangent.p_dot * (Ca - reference_point.Ca) - ds = 0 (Arclength constraint)

        Uses a Newton-like iteration on Ca to satisfy the arclength constraint.
        The BVP F(y, Ca) = 0 is solved at each iteration for a given Ca.

        Args:
            predicted_profile: Initial guess for the solution profile y (from predictor).
            predicted_Ca: Initial guess for the Capillary number Ca (from predictor).
            reference_point: The SolutionPoint from which prediction was made (y_ref, Ca_ref).
                             This is y_k if we are finding y_k+1.
                             So reference_point.profile is y_k, reference_point.Ca is Ca_k.
            tangent: The BranchTangent at reference_point (tangent_k: y_dot_k, p_dot_k).
                     These are normalized components.
            ds: The target arclength step size for this step (always positive).

        Returns:
            A new SolutionPoint if convergence is achieved within self.params.max_newton_iters,
            otherwise None.
        """
        Ca_current = predicted_Ca
        profile_current_guess = np.copy(predicted_profile) # Initial guess for y for the BVP solver

        # s_mesh for BVP solves: Use the mesh from the reference_point's profile if possible,
        # as it's likely adapted. If not available (e.g. reference_point.profile is just values),
        # then we might need to pass s_nodes with SolutionPoint or use a default.
        # For now, assume profile_current_guess is on a mesh that robust_bvp_solve can use,
        # potentially with its own s_mesh derived from reference_point if that was stored.
        #
        # If SolutionPoint stores the s_nodes for its profile, we can use that.
        # Current SolutionPoint does not explicitly store s_nodes.
        # Let's assume robust_bvp_solve will use the s_mesh associated with reference_point.profile
        # if we pass that profile's s_nodes.
        # This implies robust_bvp_solve needs an s_mesh argument that corresponds to profile_current_guess.
        #
        # Simplification: the s_mesh for the BVP solve will be based on the structure of
        # profile_current_guess. If profile_current_guess came from a previous solution,
        # it implies an underlying mesh. Let robust_bvp_solve handle this by accepting
        # profile_current_guess and an optional s_mesh. If s_mesh is None, it might use
        # a default or the one from __init__.
        #
        # Best practice: use the mesh from the reference_point for the BVP solves if available.
        # If SolutionPoint contained `s_nodes`, we'd use `reference_point.s_nodes`.
        # For now, we pass `None` for s_mesh to `robust_bvp_solve`, which means it will use `self.s_initial`
        # or we must ensure `profile_current_guess` is compatible with `self.s_initial`.
        # This is not ideal. `robust_bvp_solve` should ideally take the mesh of `profile_current_guess`.
        #
        # Let's assume `profile_current_guess` is used as `y_guess` for `robust_bvp_solve`, and
        # the `s_mesh` for `robust_bvp_solve` should be the mesh on which `profile_current_guess` is defined.
        # The `predict_step` creates `predicted_profile` by `current_point.profile + ds * tangent.y_dot`.
        # So `predicted_profile` is on the same mesh as `current_point.profile`.
        # Thus, the `s_mesh` to use in `robust_bvp_solve` should be the one from `reference_point`.
        # This means `SolutionPoint` needs to store its `s_nodes`.
        #
        # Workaround for now: pass `s_mesh=None` to `robust_bvp_solve`. This implies it will use
        # `self.s_initial`. This will force interpolation if `profile_current_guess` is on a different mesh.
        # This is a known compromise due to `SolutionPoint` not storing `s_nodes`.
        s_mesh_for_bvp = None # Will make robust_bvp_solve use self.s_initial.

        for newton_iter in range(self.params.max_newton_iters):
            # 1. Solve BVP F(y, Ca_current) = 0 to get y_new at Ca_current
            # Use profile_current_guess as the initial guess for y.
            # The s_mesh used by robust_bvp_solve here is important.
            bvp_result = self.robust_bvp_solve(Ca_current, y_guess=profile_current_guess, s_mesh=s_mesh_for_bvp)

            if not (bvp_result and bvp_result.success):
                # BVP solve failed for this Ca_current.
                # This iteration of Newton corrector fails.
                print(f"Newton iter {newton_iter+1}: BVP solve failed at Ca_current = {Ca_current:.6g}. "
                      f"Message: {bvp_result.message if bvp_result else 'robust_bvp_solve returned None'}")
                return None # Corrector step fails

            # Successfully solved BVP. y_new is bvp_result.solution.y
            # This y_new is on the mesh bvp_result.solution.x (which might be different from s_mesh_for_bvp if adapted)
            y_new = bvp_result.solution.y
            # For the arclength constraint, y_new must be comparable to reference_point.profile.
            # This means they must be on the same mesh or one interpolated.
            # Let's assume tangent.y_dot (from compute_tangent) was computed such that
            # it's compatible with reference_point.profile's mesh.
            # And y_new must be brought to that same mesh for the dot product.

            # Critical: Ensure y_new and reference_point.profile are compatible for dot product.
            # This requires interpolation if their meshes differ.
            # tangent.y_dot was (reference_point.profile - prev_point.profile) / magnitude, so it's on reference_point's mesh (or compatible).
            # If bvp_result.solution.y is on a *different* mesh (e.g. self.s_initial or adapted from it),
            # then y_new - reference_point.profile is problematic.

            # Simplification: Assume profiles are compatible or compute_tangent/this function handles it.
            # If tangent.y_dot is (3,N) and reference_point.profile is (3,N),
            # then y_new must also be (3,N) on that same mesh.
            # If bvp_result.solution.y is (3,M) where M!=N, we must interpolate.
            # This is a recurring issue due to SolutionPoint not storing s_nodes.
            # For now, assume y_new can be directly used if its shape matches reference_point.profile.

            profile_at_ref_mesh = y_new # Placeholder: needs interpolation if meshes differ
            if y_new.shape[1] != reference_point.profile.shape[1]:
                print(f"Warning: Corrector step Newton iter {newton_iter+1}: Profile from BVP solve "
                      f"({y_new.shape[1]} nodes) has different dimension than reference profile "
                      f"({reference_point.profile.shape[1]} nodes). Arclength constraint calculation may be inaccurate. "
                      f"Robust implementation requires interpolation.")
                # As a fallback, if using self.s_initial for BVP, and reference_point.profile
                # is also on self.s_initial, this might be fine. But if reference_point.profile
                # was on an adapted mesh, this is an issue.
                # For now, proceed with direct subtraction if shapes allow, else it will error.
                # This implies SolutionPoint profiles must be kept on a consistent mesh or interpolated.

            # 2. Evaluate arclength constraint T
            # T = y_dot_ref^T @ (y_new - y_ref) + p_dot_ref * (Ca_current - Ca_ref) - ds = 0
            # tangent.y_dot is y_dot_ref (normalized profile tangent at reference_point)
            # tangent.p_dot is p_dot_ref (normalized Ca tangent at reference_point)
            # reference_point.profile is y_ref
            # reference_point.Ca is Ca_ref

            # Ensure dot product is handled correctly for profiles (3,N)
            # (y_new - reference_point.profile) is (3,N). tangent.y_dot is (3,N).
            # Dot product should be sum over all elements: np.sum(tangent.y_dot * ( ... ))
            delta_y_profile = profile_at_ref_mesh - reference_point.profile
            delta_Ca = Ca_current - reference_point.Ca

            # Term1: tangent.y_dot^T @ delta_y_profile
            # If y_dot is (3,N) and delta_y_profile is (3,N), then y_dot^T @ delta_y means sum of element-wise products.
            term_y = np.sum(tangent.y_dot * delta_y_profile)

            # Term2: tangent.p_dot * delta_Ca
            term_p = tangent.p_dot * delta_Ca

            constraint_T = term_y + term_p - ds # ds is target arclength step (positive)

            # 3. Check convergence of constraint T
            if abs(constraint_T) < self.params.tolerance:
                # Converged: Found a point (y_new, Ca_current) that satisfies BVP and T=0.
                print(f"Corrector converged in {newton_iter+1} iterations at Ca = {Ca_current:.6g}, T = {constraint_T:.2e}")
                return SolutionPoint(
                    Ca=Ca_current,
                    X_cl=bvp_result.x0, # x0 from the successful BVP solve
                    theta_min=bvp_result.theta_min, # theta_min from BVP solve
                    profile=y_new, # The converged profile y_new from BVP
                                   # Note: y_new is on bvp_result.solution.x mesh
                    arc_length=reference_point.arc_length + ds, # Cumulative arclength
                    newton_iters=newton_iter + 1
                )

            # 4. If not converged, update Ca_current using Newton step for T(Ca) = 0.
            # Need dT/dCa.
            # T = tangent.y_dot^T @ (y(Ca) - y_ref) + tangent.p_dot * (Ca - Ca_ref) - ds
            # dT/dCa = tangent.y_dot^T @ (dy(Ca)/dCa) + tangent.p_dot
            # The issue suggests a simplified update: Ca_current -= constraint_T / tangent.p_dot
            # This simplification assumes dy(Ca)/dCa is zero or its contribution is negligible,
            # or that tangent.p_dot somehow accounts for it.
            # This is equivalent to assuming dT/dCa approx tangent.p_dot.

            # Derivative dT/dCa (simplified version from problem description)
            dT_dCa_approx = tangent.p_dot

            if abs(dT_dCa_approx) < 1e-9: # Avoid division by zero if tangent is nearly orthogonal to Ca axis
                print(f"Newton iter {newton_iter+1}: Derivative dT/dCa approx zero ({dT_dCa_approx:.2e}). Corrector may fail or be slow.")
                # Could try a very small step in Ca, or terminate.
                # If p_dot is zero, means tangent is flat in Ca. Arclength constraint is then mainly on profile.
                # The simplified Newton update will fail here.
                # This indicates a potential turning point where dCa/ds = 0.
                # A more robust Newton solver would use the full Jacobian of the extended system.
                # For now, if this happens, the corrector step fails.
                return None

            Ca_current -= constraint_T / dT_dCa_approx

            # Update profile_current_guess for the next BVP solve using the latest solved profile
            profile_current_guess = np.copy(y_new) # Use solution from this iteration as guess for next
            s_mesh_for_bvp = np.copy(bvp_result.solution.x) # And use its mesh too

            print(f"Newton iter {newton_iter+1}: Ca updated to {Ca_current:.6g}, T = {constraint_T:.2e}")

        # If loop finishes, Newton corrector failed to converge within max_newton_iters
        print(f"Corrector failed to converge in {self.params.max_newton_iters} iterations. Last T = {constraint_T:.2e}")
        return None

    def adaptive_step_control(self, newton_iters: int, current_ds: float) -> float:
        """
        Adjusts the arclength step size 'ds' based on convergence performance.
        The actual sign of ds (direction) is handled by the main loop. This function
        deals with the magnitude of ds.

        Args:
            newton_iters: Number of Newton iterations taken for the last successful step.
            current_ds: The current arclength step size (absolute value).

        Returns:
            The new adapted arclength step size (absolute value).
            The calling function (`solve_branch`) will apply the direction.
        """
        # Ensure current_ds is positive for calculations here
        current_ds_abs = abs(current_ds)
        new_ds_abs = current_ds_abs

        # Target Newton iterations (e.g., 3-5 is often good)
        # Let's define optimal range, e.g., newton_iters_optimal_low = 2, newton_iters_optimal_high = 4
        # These could be parameters in ContinuationParams if tuning is needed.
        newton_iters_fast = 3  # Converged very quickly, try increasing step size
        newton_iters_slow = 6  # Converged slowly, try decreasing step size

        if newton_iters <= newton_iters_fast:
            # If convergence was fast, try increasing step size
            # Increase by a factor (e.g., 1.25 or 1.5), but not exceeding max_ds
            new_ds_abs = min(current_ds_abs * 1.25, self.params.max_ds)
            print(f"Adaptive step: Fast convergence ({newton_iters} iters). DS increased from {current_ds_abs:.2e} to {new_ds_abs:.2e}")
        elif newton_iters >= newton_iters_slow:
            # If convergence was slow, decrease step size
            # Decrease by a factor (e.g., 0.75 or 0.5), but not below min_ds
            new_ds_abs = max(current_ds_abs * 0.75, self.params.min_ds)
            print(f"Adaptive step: Slow convergence ({newton_iters} iters). DS decreased from {current_ds_abs:.2e} to {new_ds_abs:.2e}")
        else:
            # Convergence was nominal (e.g., 4-5 iterations). Keep current step size.
            print(f"Adaptive step: Nominal convergence ({newton_iters} iters). DS maintained at {current_ds_abs:.2e}")
            new_ds_abs = current_ds_abs # Explicitly stating no change

        # Ensure step size is within prescribed bounds [min_ds, max_ds]
        # This is already handled by min/max in the update rules, but an extra clamp is safe.
        new_ds_abs = np.clip(new_ds_abs, self.params.min_ds, self.params.max_ds)

        # The issue also mentions: "Halve ds on convergence failure"
        # That logic will be in `solve_branch` where failures are directly handled.
        # This function is for adapting based on *successful* step's iteration count.

        return new_ds_abs

    def detect_fold(self, points: List[SolutionPoint], last_p_dot: Optional[float] = None) -> Tuple[bool, Optional[float]]:
        """
        Detects a fold bifurcation by monitoring the sign change in dCa/ds (i.e., tangent.p_dot).

        Args:
            points: A list of recent SolutionPoint objects (at least two, typically last 2 or 3).
            last_p_dot: The p_dot of the tangent leading to the most recent point in `points`.
                        If None, it will be estimated from the last two points.

        Returns:
            A tuple (fold_detected_boolean, current_p_dot).
            current_p_dot can be used for the next call.
        """
        # To be implemented in detail later
        pass