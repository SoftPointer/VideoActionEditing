"""Self-Predicted Phase-Transport (SPT) Bernini prototype.

This package is intentionally separate from the CDF-v1 implementation.  The
runtime contract is source-video + instruction only.  Paired target latents
are accepted by :func:`build_oracle_plan` during training/evaluation only.
"""

from .phase_transport import (  # noqa: F401
    GATE_GENERATE,
    GATE_PRESERVE,
    GATE_TRANSPORT,
    LATENT_PHASES,
    PhasePlan,
    PhaseTransportAdapter,
    PhaseTransportConfig,
    PhaseTransportError,
    build_oracle_plan,
    execute_clean_plan,
    execute_packed_velocity,
    exact_identity_plan,
    make_proxy_target,
    velocity_from_clean,
)
from .phase_query_planner import (  # noqa: F401
    ARCHITECTURE_NAME as PHASE_QUERY_ARCHITECTURE,
    PhaseQueryPlanner,
    PhaseQueryPlannerConfig,
    PhaseQueryPlannerError,
    normalized_position_channels,
    sinusoidal_phase_encoding,
)
