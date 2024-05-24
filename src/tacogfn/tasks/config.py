"""
Task config from
https://github.com/recursionpharma/gflownet
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SEHTaskConfig:
    pass  # SEH just uses a temperature conditional


@dataclass
class UniDockFinetuneTaskConfig:
    pocket_index: int = 0
    unidock_mode: str = "balance"


@dataclass
class PharmacoFragTaskConfig:
    """Config for the PharmacoFragTask

    Attributes
    ----------
    fragment_type : str
        The type of fragment to use
    """

    fragment_type: str = "gflownet"  # or zinc250k_50cutoff_brics
    affinity_predictor: str = "beta"  # or "alpha"
    docking_score_sigmoid: bool = False
    docking_score_exp: float = 1.0
    max_qed_reward: float = 0.7  # no extra reward for qed above this
    max_sa_reward: float = 0.75  # no extra reward for sa below this
    mol_adj: float = 0  # ds / num_atoms^mol_adj
    reward_multiplier: float = 1.0
    ablation: str = "none"  # [no_pharmaco, pocket_graph]
    objectives: List[str] = field(default_factory=lambda: ["docking", "qed", "sa"])


@dataclass
class PocketMOOTaskConfig:
    use_steer_thermometer: bool = False
    preference_type: Optional[str] = "dirichlet"
    focus_type: Optional[str] = None
    focus_dirs_listed: Optional[List[List[float]]] = None
    focus_cosim: float = 0.0
    focus_limit_coef: float = 1.0
    focus_model_training_limits: Optional[Tuple[int, int]] = None
    focus_model_state_space_res: Optional[int] = None
    max_train_it: Optional[int] = None
    n_valid: int = 15
    n_valid_repeats: int = 128
    objectives: List[str] = field(default_factory=lambda: ["seh", "qed", "sa", "mw"])


@dataclass
class SEHMOOTaskConfig:
    """Config for the SEHMOOTask

    Attributes
    ----------
    use_steer_thermometer : bool
        Whether to use a thermometer encoding for the steering.
    preference_type : Optional[str]
        The preference sampling distribution, defaults to "dirichlet".
    focus_type : Union[list, str, None]
        The type of focus distribtuion used, see SEHMOOTask.setup_focus_regions.
    focus_cosim : float
        The cosine similarity threshold for the focus distribution.
    focus_limit_coef : float
        The smoothing coefficient for the focus reward.
    focus_model_training_limits : Optional[Tuple[int, int]]
        The training limits for the focus sampling model (if used).
    focus_model_state_space_res : Optional[int]
        The state space resolution for the focus sampling model (if used).
    max_train_it : Optional[int]
        The maximum number of training iterations for the focus sampling model (if used).
    n_valid : int
        The number of valid cond_info tensors to sample
    n_valid_repeats : int
        The number of times to repeat the valid cond_info tensors
    objectives : List[str]
        The objectives to use for the multi-objective optimization. Should be a subset of ["seh", "qed", "sa", "wt"].
    """

    use_steer_thermometer: bool = False
    preference_type: Optional[str] = "dirichlet"
    focus_type: Optional[str] = None
    focus_dirs_listed: Optional[List[List[float]]] = None
    focus_cosim: float = 0.0
    focus_limit_coef: float = 1.0
    focus_model_training_limits: Optional[Tuple[int, int]] = None
    focus_model_state_space_res: Optional[int] = None
    max_train_it: Optional[int] = None
    n_valid: int = 15
    n_valid_repeats: int = 128
    objectives: List[str] = field(default_factory=lambda: ["seh", "qed", "sa", "mw"])


@dataclass
class TasksConfig:
    seh: SEHTaskConfig = SEHTaskConfig()
    seh_moo: SEHMOOTaskConfig = SEHMOOTaskConfig()
    pocket_moo: PocketMOOTaskConfig = PocketMOOTaskConfig()
    pharmaco_frag: PharmacoFragTaskConfig = PharmacoFragTaskConfig()
    finetune: UniDockFinetuneTaskConfig = UniDockFinetuneTaskConfig()
