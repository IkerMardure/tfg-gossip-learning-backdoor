"""
Centralized logging module for GLow project.
Provides verbosity control for different log levels across the project.
"""

import os
from typing import Literal

LogLevel = Literal["minimal", "standard", "verbose"]

# Global configuration dict (set by config on startup)
_log_config: dict = {
    "client_training": "standard",
    "data_poisoning": "standard",
    "pretraining": "standard",
    "results": "standard",
    "heartbeat": "standard",
}

_VALID_LEVELS = {"minimal", "standard", "verbose"}


def _normalize_level(level: str, fallback: str = "standard") -> str:
    level_str = str(level).strip().lower()
    return level_str if level_str in _VALID_LEVELS else fallback


def _configure_from_environment() -> None:
    """Allow configuring verbosity for helper scripts that do not load YAML directly."""
    env_level = os.environ.get("GLOW_VERBOSE_LOGGING")
    if not env_level:
        return
    normalized = _normalize_level(env_level)
    for key in _log_config:
        _log_config[key] = normalized


def configure_logging(config: dict) -> None:
    """
    Configure logging levels from YAML config.
    
    Args:
        config: Dict with keys like 'verbose_logging' (level) or 'log_level_<component>'
    """
    global _log_config
    
    # Set global default (if present)
    if "verbose_logging" in config:
        default_level = _normalize_level(config["verbose_logging"])
        for key in _log_config:
            _log_config[key] = default_level
    
    # Override per-component if specified
    for key in _log_config:
        config_key = f"log_level_{key}"
        if config_key in config:
            _log_config[key] = _normalize_level(config[config_key], fallback=_log_config[key])


def log(component: str, message: str, level: LogLevel = "standard") -> None:
    """
    Log a message if the configured level allows it.
    
    Args:
        component: Name of component (e.g., 'client_training', 'data_poisoning')
        message: Message to log
        level: Log level of this message ('minimal', 'standard', 'verbose')
    """
    configured_level = _log_config.get(component, "standard")
    
    # Decide whether to print based on verbosity hierarchy:
    # minimal < standard < verbose
    level_order = {"minimal": 0, "standard": 1, "verbose": 2}
    msg_importance = level_order.get(level, 1)
    config_importance = level_order.get(configured_level, 1)
    
    if msg_importance <= config_importance:
        print(message)


def log_client_training(message: str, level: LogLevel = "standard") -> None:
    """Log client training messages."""
    log("client_training", message, level)


def log_data_poisoning(message: str, level: LogLevel = "standard") -> None:
    """Log data poisoning messages."""
    log("data_poisoning", message, level)


def log_pretraining(message: str, level: LogLevel = "standard") -> None:
    """Log pretraining phase messages."""
    log("pretraining", message, level)


def log_results(message: str, level: LogLevel = "standard") -> None:
    """Log results/metrics messages."""
    log("results", message, level)


def log_heartbeat(message: str, level: LogLevel = "standard") -> None:
    """Log heartbeat status messages."""
    log("heartbeat", message, level)


_configure_from_environment()
