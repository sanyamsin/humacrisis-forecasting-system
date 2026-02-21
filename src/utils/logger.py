from loguru import logger
import sys
import os

def setup_logger(log_level: str = "INFO"):
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | {message}",
        level=log_level,
        colorize=True
    )
    logger.add(
        "logs/humacrisis.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG"
    )
    return logger

log = setup_logger()