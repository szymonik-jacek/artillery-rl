"""
Classical FCS firing rules used as baselines for the DQN agent.

Each policy decides FIRE/HOLD from the environment state at the current
decision step and, unlike the DQN agent, reads the hit probability (HP) that
the fire-control chain computes at that step.
"""

from rl_training.environment import Action


class ThresholdPolicy:
    """Pure threshold rule: FIRE at the first step where P_hit >= p_thr."""

    def __init__(self, p_thr: float):
        if not 0.0 <= p_thr <= 1.0:
            raise ValueError(f"p_thr must be in [0, 1], got {p_thr}")
        self.p_thr = p_thr

    @property
    def name(self) -> str:
        return f"Threshold_{self.p_thr:.2f}"

    def act(self, state) -> Action:
        if state.current_hit_probability >= self.p_thr:
            return Action.FIRE
        return Action.HOLD


class DecayingThresholdPolicy:
    """Threshold that falls linearly from p_start to p_end over the engagement window.

    The elapsed fraction of the window is 1 - time_remaining / tracking_duration,
    so the rule uses only quantities a real FCS knows at the decision step.
    """

    def __init__(self, p_start: float, p_end: float):
        if not 0.0 <= p_end <= p_start <= 1.0:
            raise ValueError(f"Need 0 <= p_end <= p_start <= 1, got {p_start}, {p_end}")
        self.p_start = p_start
        self.p_end = p_end

    @property
    def name(self) -> str:
        return f"Decaying_{self.p_start:.2f}_{self.p_end:.2f}"

    def threshold(self, state) -> float:
        elapsed = 1.0 - state.time_remaining / state.scenario.tracking_duration
        elapsed = min(1.0, max(0.0, elapsed))
        return self.p_start + (self.p_end - self.p_start) * elapsed

    def act(self, state) -> Action:
        if state.current_hit_probability >= self.threshold(state):
            return Action.FIRE
        return Action.HOLD


class CovarianceGatedThresholdPolicy(ThresholdPolicy):
    """Threshold rule gated on Kalman convergence.

    FIRE when P_hit >= p_thr and Tr(P_t) <= tau_cov. tau_cov is in the units
    of the raw (unclamped) covariance trace stored in the environment state.
    """

    def __init__(self, p_thr: float, tau_cov: float):
        super().__init__(p_thr)
        if tau_cov <= 0.0:
            raise ValueError(f"tau_cov must be positive, got {tau_cov}")
        self.tau_cov = tau_cov

    @property
    def name(self) -> str:
        return f"CovGated_{self.p_thr:.2f}_{self.tau_cov:g}"

    def act(self, state) -> Action:
        if state.covariance_trace > self.tau_cov:
            return Action.HOLD
        return super().act(state)


class PlateauPolicy:
    """Fire when HP has stopped rising.

    FIRE at the first step where P_hit >= min_hp and P_hit rose by less than
    eps over the last `window` steps. The policy keeps its own HP history,
    reset at each episode's first step.
    """

    def __init__(self, min_hp: float, window: int, eps: float):
        if not 0.0 <= min_hp <= 1.0:
            raise ValueError(f"min_hp must be in [0, 1], got {min_hp}")
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.min_hp = min_hp
        self.window = window
        self.eps = eps
        self._history = []

    @property
    def name(self) -> str:
        return f"Plateau_{self.min_hp:.2f}_k{self.window}_e{self.eps:g}"

    def act(self, state) -> Action:
        if state.episode_step == 0:
            self._history = []
        hp = state.current_hit_probability
        self._history.append(hp)
        if hp < self.min_hp or len(self._history) <= self.window:
            return Action.HOLD
        if hp - self._history[-1 - self.window] < self.eps:
            return Action.FIRE
        return Action.HOLD
