"""
Backend Configuration.
"""

from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Simulation limits
    max_resolution: int = 2000
    max_concurrent_tasks: int = 2

    # Internal parameters (not exposed to user)
    default_pixel_size: float = 1e-6
    padding_ratio: float = 0.1  # ASM/SFR padding to prevent edge artifacts
    progress_report_interval: int = 50  # Report progress every N iterations

    # Task management
    task_timeout_seconds: int = 3600  # 1 hour max per task
    task_cleanup_seconds: int = 86400  # Clean up completed tasks after 24 hours


# Global config instance
config = AppConfig()
