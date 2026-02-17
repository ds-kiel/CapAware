import torch
import torch.nn as nn
from typing import Optional

class QuantileLoss(torch.nn.Module):
    def __init__(self, quantile=0.45):
        """
        Quantile Loss implementation in PyTorch.
        :param quantile: The quantile to prioritize (e.g., 0.1 for 10% quantile).
        """
        super(QuantileLoss, self).__init__()
        assert 0 < quantile < 1, "Quantile must be between 0 and 1."
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        """
        Compute the quantile loss.
        :param y_pred: Predicted values (torch.Tensor).
        :param y_true: Ground truth values (torch.Tensor).
        :return: Quantile loss (torch.Tensor).
        """
        error = y_true - y_pred
        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)
        return torch.mean(loss)
    
class CustomWeightedLoss(torch.nn.Module):
    def __init__(self, lambda_penalty=2.0):
        """
        Custom Weighted Loss for penalizing over-predictions.
        :param lambda_penalty: Multiplier for penalizing over-predictions.
        """
        super(CustomWeightedLoss, self).__init__()
        self.lambda_penalty = lambda_penalty
        self.mse = torch.nn.MSELoss()  # Base loss function (MSE)

    def forward(self, y_pred, y_true):
        """
        Compute the custom weighted loss.
        :param y_pred: Predicted values (torch.Tensor).
        :param y_true: Ground truth values (torch.Tensor).
        :return: Custom weighted loss (torch.Tensor).
        """
        mse_loss = self.mse(y_pred, y_true)  # Base MSE loss
        # Over-prediction penalty
        penalty = torch.sum(torch.clamp(y_pred - y_true, min=0))
        #penalty = torch.mean(torch.clamp(y_pred - y_true, min=0))  # Mean over batch
        return mse_loss + self.lambda_penalty * penalty
    
class SmartProvisionLoss(torch.nn.Module):
    def __init__(self, alpha, beta, epsilon_beta):
        """
        Custom loss function to discourage overprovisioning while allowing slight underprovisioning.

        Parameters:
        - alpha: Small penalty for underprovisioning (default 0.5).
        - beta: Strong penalty for overprovisioning (default 5.0).
        - epsilon_beta: Small tolerance for underprovisioning before penalty applies (default 0.1).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon_beta = epsilon_beta

    def forward(self, pred, target):
        error = pred - target  # Positive = overprovisioning, Negative = underprovisioning
        
        # Allow small underprovisioning within epsilon_beta without penalty
        # Mild linear penalty for underprovisioning to prevent resource waste
        underprovisioning_loss = torch.where(error < -self.epsilon_beta, self.alpha * torch.abs(error), 0.0)
        
        # Overprovisioning is strictly penalized
        overprovisioning_loss = torch.where(error > 0, self.beta * error, 0.0)

        loss = underprovisioning_loss + overprovisioning_loss
        return loss.mean()  # Mean loss over batch
    
class ARULoss(torch.nn.Module):
    """
    Asymmetric Relative Utilization (ARU) Loss

    Penalizes predictions relative to network capacity with three regimes:
      1. Overutilization (r > 1): strong quadratic penalty (penalty_over)
      2. Tolerable underutilization (underutil_threshold <= r <= 1): small linear penalty (penalty_mild)
      3. Deep underutilization (r < underutil_threshold): moderate linear penalty (penalty_deep)

    Loss is computed per-element and then averaged using only tensor clamps.

    Parameters:
        penalty_over (float): Quadratic penalty factor for overutilization (r > 1).
        penalty_mild (float): Linear penalty factor for tolerable underutilization (underutil_threshold <= r <= 1).
        penalty_deep (float): Linear penalty factor for deep underutilization (r < underutil_threshold).
        underutil_threshold (float): Utilization ratio cutoff distinguishing tolerable vs. deep underutilization (0 < underutil_threshold < 1).
    """
    def __init__(
        self,
        penalty_over: float,
        penalty_mild: float,
        penalty_deep: float,
        underutil_threshold: float,
        ):
        super().__init__()
        assert penalty_over >= 0,   "penalty_over must be non-negative"
        assert penalty_mild >= 0,   "penalty_mild must be non-negative"
        assert penalty_deep >= 0,   "penalty_deep must be non-negative"
        assert 0 < underutil_threshold < 1, "underutil_threshold must be between 0 and 1"

        self.penalty_over = penalty_over
        self.penalty_mild = penalty_mild
        self.penalty_deep = penalty_deep
        self.underutil_threshold = underutil_threshold
        self._tol_inv = 1 - underutil_threshold

    def forward(self, pred: torch.Tensor, capacity: torch.Tensor) -> torch.Tensor:
        """
        Compute the Asymmetric Relative Utilization Loss without explicit masks.

        Args:
            pred (Tensor): Predicted bandwidth usage.
            capacity (Tensor): Network capacity (same shape as pred).

        Returns:
            Tensor: Averaged loss (scalar).
        """
        # Relative utilization
        r = pred / capacity

        # 1) Overutilization: (r - 1) clamped at zero
        over = torch.clamp(r - 1.0, min=0.0)
        loss_over = self.penalty_over * over.pow(2)

        # 2 & 3) Underutilization: base clamp then split
        base_under = torch.clamp(1.0 - r, min=0.0)
        mild = torch.clamp(base_under, max=self._tol_inv)
        deep = base_under - mild
        loss_mild = self.penalty_mild * mild
        loss_deep = self.penalty_deep * deep

        # Combine and average
        return (loss_over + loss_mild + loss_deep).mean()
    
class HybridARULoss(nn.Module):
    """
    Hybrid Asymmetric Relative Utilization (ARU) Loss

    Region selection (ratio-based):
        • r = pred / capacity
        • r > 1                     → over-utilisation  (quadratic penalty on abs error)
        • underutil_threshold ≤ r ≤ 1 → mild under-util (linear penalty on abs error)
        • r < underutil_threshold   → deep under-util  (linear penalty on abs error)

    Penalty magnitude (absolute):
        err = pred - capacity

    Args
    ----
    penalty_over         - weight for quadratic over-prediction penalty
    penalty_mild         - weight for linear mild-under-prediction penalty
    penalty_deep         - weight for linear deep-under-prediction penalty
    underutil_threshold  - boundary between mild and deep under-util (0 < T < 1)
    """
    def __init__(
            self,
            penalty_over: float = 6.0,
            penalty_mild: float = 0.5,
            penalty_deep: float = 0.75,
            underutil_threshold: float = 0.90,
            exponent_over: float = 2.0, # 2.0 for quadratic, 3.0 for cubic
    ):
        super().__init__()
        assert 0 < underutil_threshold < 1, "underutil_threshold must lie in (0, 1)"
        for name, val in dict(penalty_over=penalty_over,
                              penalty_mild=penalty_mild,
                              penalty_deep=penalty_deep).items():
            assert val >= 0, f"{name} must be non-negative"

        self.penalty_over = penalty_over
        self.penalty_mild = penalty_mild
        self.penalty_deep = penalty_deep
        self.underutil_threshold = underutil_threshold
        self.exponent_over = exponent_over
        self.eps : float = 1e-6

    def forward(self, 
                prediction: torch.Tensor, 
                capacity: torch.Tensor) -> torch.Tensor:
        r   = prediction / (capacity + self.eps)       # relative utilisation
        err = prediction - capacity                    # absolute error

        # region masks
        over_mask  =  (r > 1).float()
        mild_mask  = ((r <= 1) & (r >= self.underutil_threshold)).float()
        deep_mask  =  (r < self.underutil_threshold).float()

        # ── 1) over-util loss
        overutil = torch.clamp(err,  min=0.0)
        loss_over =  self.penalty_over * (overutil.pow(self.exponent_over)) * over_mask

        # ── 2) under-util losses
        underutil = torch.clamp(-err, min=0.0)
        loss_mild = self.penalty_mild * underutil * mild_mask
        loss_deep = self.penalty_deep * underutil * deep_mask

        return (loss_over + loss_mild + loss_deep).mean()
    
class ARULossHO(nn.Module):
    """
    Asymmetric Relative Utilisation Loss with HO-aware softening
    ------------------------------------------------------------
    If `ho_prob` is **omitted**, the loss behaves exactly like the
    original ARU loss (i.e. assumes no hand-overs).

    Parameters
    ----------
    penalty_over         : weight for quadratic over-prediction penalty
    penalty_mild         : weight for linear mild under-prediction penalty
    penalty_deep         : weight for linear deep under-prediction penalty
    underutil_threshold  : boundary between mild & deep under-util (0 < T < 1)
    exponent_over        : 2 = quadratic, 3 = cubic over-penalty ...
    soft_factor          : how strongly to relax under-penalties when ho_prob→1
    """

    def __init__(
            self,
            penalty_over: float = 6.0,
            penalty_mild: float = 0.5,
            penalty_deep: float = 0.75,
            underutil_threshold: float = 0.90,
            exponent_over: float = 2.0,
            soft_factor: float = 0.5,
    ):
        super().__init__()

        assert 0 < underutil_threshold < 1, "underutil_threshold must lie in (0,1)"
        for name, val in dict(penalty_over=penalty_over,
                              penalty_mild=penalty_mild,
                              penalty_deep=penalty_deep,
                              soft_factor=soft_factor).items():
            assert val >= 0, f"{name} must be non-negative"
        assert soft_factor <= 1, "soft_factor should be ≤ 1"

        self.penalty_over  = penalty_over
        self.penalty_mild  = penalty_mild
        self.penalty_deep  = penalty_deep
        self.underutil_threshold = underutil_threshold
        self.exponent_over = exponent_over
        self.soft_factor   = soft_factor
        self.eps: float    = 1e-6

    # ------------------------------------------------------------------ #

    def forward(self,
                prediction: torch.Tensor,
                capacity:   torch.Tensor,
                ho_prob:    Optional[torch.Tensor] = None) -> torch.Tensor:
        r   = prediction / (capacity + self.eps)
        err = prediction - capacity
        
        if ho_prob is None:
            ho_prob = prediction.new_zeros(prediction.shape)

        # dynamic under-prediction weights
        penalty_mild_eff = self.penalty_mild# * (1 - self.soft_factor * ho_prob)
        penalty_deep_eff = self.penalty_deep# * (1 - self.soft_factor * ho_prob)

        #alpha = 1.0  # extra weight when ho_prob == 1
        penalty_over_eff = self.penalty_over * (1 + self.soft_factor * ho_prob)

        # region masks
        over_mask  =  (r > 1).float()
        mild_mask  = ((r <= 1) & (r >= self.underutil_threshold)).float()
        deep_mask  =  (r <  self.underutil_threshold).float()

        # ── 1) over-util loss
        overutil   = torch.clamp(err,  min=0.0)
        loss_over  = penalty_over_eff * overutil.pow(self.exponent_over) * over_mask

        # ── 2) under-util losses
        underutil  = torch.clamp(-err, min=0.0)
        loss_mild  = penalty_mild_eff * underutil * mild_mask
        loss_deep  = penalty_deep_eff * underutil * deep_mask

        return (loss_over + loss_mild + loss_deep).mean()