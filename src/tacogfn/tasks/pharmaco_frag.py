from src.tacogfn.data.pharmacophore import PharmacophoreGraphDataset
from src.tacogfn.trainers import FlatRewards, GFNTask, RewardScalar
from src.tacogfn.utils.conditioning import TemperatureConditional


class PharmacophoreTask(GFNTask):
    """
    - Non Multi-Objective
    - Pharmacophore represented as an embedding vector
    - Vanilla fragment based molecular construction environment
    """

    def __init__(
        self,
        dataset: Dataset,
        pharmacophore_dataset: PharmacophoreGraphDataset,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def _load_task_models(self):
        # TODO: change this to affinity model KAIST is developing
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model, send_to_device=True)
        return {"affinity": model}
