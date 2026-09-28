"""rlm learning loop — mines traces, evolves agent configs.

Wraps the runtime's LearningOrchestrator. Call `run_cycle()` manually or
schedule it periodically (e.g. every N interactions) from system.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

from openjarvis.learning.learning_orchestrator import LearningOrchestrator

from ares.observability import get_trace_store

logger = logging.getLogger(__name__)

# Agent configs Ares produces will be written here for inspection
ARES_CONFIG_DIR = Path.home() / ".ares" / "evolved_configs"


def build_learning_orchestrator() -> LearningOrchestrator:
    ARES_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return LearningOrchestrator(
        trace_store=get_trace_store(),
        config_dir=ARES_CONFIG_DIR,
        min_improvement=0.02,
        min_sft_pairs=5,       # low threshold — small local dataset
        min_quality=0.6,       # accept traces with feedback >= 0.6
    )


def run_cycle(orchestrator: LearningOrchestrator, agent_id: str | None = None) -> dict:
    """Run one trace→mine→evolve cycle. Returns a summary dict."""
    logger.info("Starting rlm learning cycle (agent=%s)", agent_id or "all")
    result = orchestrator.run(agent_id=agent_id)
    logger.info("rlm cycle complete: %s", result)
    return result
