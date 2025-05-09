"""Main entry point for the Stock-Sentiment module.

This script initializes the service, sets up logging, and starts
consuming messages from the configured message queue for sentiment analysis.
"""

import os
import sys

# Add 'src/' to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.setup_logger import setup_logger
from app.queue_handler import consume_messages

# Initialize logger
logger = setup_logger(__name__)


def main() -> None:
    """Starts the Sentiment Analysis Service by consuming stock data messages
    and processing sentiment signals.

    This service listens to messages from a queue (RabbitMQ or SQS),
    applies sentiment analysis, and publishes the results to a designated output.
    """
    logger.info("Starting Sentiment Analysis Service...")
    consume_messages()


if __name__ == "__main__":
    main()
