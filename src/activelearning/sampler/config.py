from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field
from activelearning.sampler.hypercube_sampler import HypercubeSampler
from activelearning.sampler.sampler import Sampler
from activelearning.runtime import RuntimeConfig


class HypercubeSamplerConfig(BaseModel):
    type: Literal["HypercubeSampler"] = "HypercubeSampler"
    bounds: list[tuple[float, float]]
    num_samples: int
    fidelities: dict[int, float] | list[int] | None = None
    point_strategy: Literal["uniform", "lhs"] = "uniform"

    def build(self, runtime: RuntimeConfig | None = None) -> Sampler:
        return HypercubeSampler(
            bounds=self.bounds,
            num_samples=self.num_samples,
            fidelities=self.fidelities,
            point_strategy=self.point_strategy,
        )


class GFlowNetSamplerConfig(BaseModel):
    type: Literal["GFlowNetSampler"] = "GFlowNetSampler"
    n_samples: int = Field(gt=0)
    device: str = "cpu"
    float_precision: Literal[32, 64] = 32
    log_dir: str | None = None
    conf: dict[str, Any] | None = None

    def _resolve_device(self, runtime: RuntimeConfig | None) -> str:
        if "device" in self.model_fields_set or runtime is None:
            return self.device
        return runtime.device

    def _resolve_float_precision(
        self, runtime: RuntimeConfig | None
    ) -> Literal[32, 64]:
        if "float_precision" in self.model_fields_set or runtime is None:
            return self.float_precision
        return runtime.precision

    def build(self, runtime: RuntimeConfig | None = None) -> Sampler:
        from activelearning.sampler.gflownet.config_utils import compose_gflownet_conf
        from activelearning.sampler.gflownet.gflownet_sampler import GFlowNetSampler

        resolved_device = self._resolve_device(runtime)
        resolved_float_precision = self._resolve_float_precision(runtime)
        full_conf = compose_gflownet_conf(
            conf_overrides=self.conf,
            log_dir=self.log_dir,
        )
        return GFlowNetSampler(
            n_samples=self.n_samples,
            conf=full_conf,
            device=resolved_device,
            float_precision=resolved_float_precision,
        )


class GFlowNetGridSamplerConfig(GFlowNetSamplerConfig):
    type: Literal["GFlowNetGridSampler"] = "GFlowNetGridSampler"  # type: ignore[assignment]
    output_bounds: list[tuple[float, float]] | None = None

    def build(self, runtime: RuntimeConfig | None = None) -> Sampler:
        from activelearning.sampler.gflownet.config_utils import compose_gflownet_conf
        from activelearning.sampler.gflownet.grid_sampler import GFlowNetGridSampler

        resolved_device = self._resolve_device(runtime)
        resolved_float_precision = self._resolve_float_precision(runtime)
        full_conf = compose_gflownet_conf(
            conf_overrides=self.conf,
            log_dir=self.log_dir,
        )
        return GFlowNetGridSampler(
            n_samples=self.n_samples,
            conf=full_conf,
            device=resolved_device,
            float_precision=resolved_float_precision,
            output_bounds=self.output_bounds,
        )


SamplerConfig = Annotated[
    Union[HypercubeSamplerConfig, GFlowNetSamplerConfig, GFlowNetGridSamplerConfig],
    Field(discriminator="type"),
]
