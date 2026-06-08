from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.config import load_config
from src.logger import setup_logging
from src.monitor import FastAPIMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastAPI ONNX inference service monitoring"
    )
    parser.add_argument(
        "--config",
        default="config/monitoring_config.yaml",
        help="Path to monitoring YAML config",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one monitoring check and exit",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    config = load_config(args.config)

    log_file = Path(config.logging.log_file)
    if not log_file.is_absolute():
        log_file = config.project_root / log_file

    logger = setup_logging(
        log_file=log_file, console_colors=config.logging.console_colors
    )
    monitor = FastAPIMonitor(config=config, logger=logger)

    if args.once:
        await monitor.maybe_start_service()
        try:
            await monitor.run_once()
        finally:
            await monitor.stop_service()
    else:
        await monitor.run_forever()


if __name__ == "__main__":
    asyncio.run(async_main())
