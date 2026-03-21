"""CLI entrypoint for the object detection agent."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Load .env from project root before anything else
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from task1_object_detection.agent.agent import ObjectDetectionAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="NM i AI - Object Detection Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Commands:
  analyze   Analyze the dataset (class distribution, bbox stats, weak categories)
  boost     Prepare dataset (COCO→YOLO conversion, oversampling, augmentation config)
  train     Train and evaluate a YOLOv8 model
  full      Run the complete pipeline: analyze → boost → train

Examples:
  python -m task1_object_detection.agent.main analyze
  python -m task1_object_detection.agent.main full
  python -m task1_object_detection.agent.main train --prompt "Train yolov8s for 100 epochs"
        """,
    )
    parser.add_argument(
        "command",
        choices=["analyze", "boost", "train", "full"],
        help="Which pipeline stage to run.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Optional custom prompt/instructions for the agent.",
    )
    args = parser.parse_args()

    agent = ObjectDetectionAgent()

    if args.command == "analyze":
        report = asyncio.run(agent.analyze(args.prompt))
        print("\n" + "=" * 60)
        print("ANALYSIS REPORT")
        print("=" * 60)
        print(report)

    elif args.command == "boost":
        report = asyncio.run(agent.boost(args.prompt))
        print("\n" + "=" * 60)
        print("BOOST REPORT")
        print("=" * 60)
        print(report)

    elif args.command == "train":
        report = asyncio.run(agent.train(args.prompt))
        print("\n" + "=" * 60)
        print("TRAINING REPORT")
        print("=" * 60)
        print(report)

    elif args.command == "full":
        reports = asyncio.run(agent.run_full_pipeline(args.prompt))
        print("\n" + "=" * 60)
        print("FULL PIPELINE RESULTS")
        print("=" * 60)
        for stage, report in reports.items():
            print(f"\n--- {stage.upper()} ---")
            print(report)

    logger.info("Done.")


if __name__ == "__main__":
    main()
