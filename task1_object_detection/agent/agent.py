"""LangGraph-based object detection agent with analyzer → booster → trainer architecture."""

import asyncio
import logging
import os
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langgraph.prebuilt import create_react_agent

from task1_object_detection.agent.config import (
    ANALYZER_TIMEOUT,
    BOOSTER_TIMEOUT,
    TRAINER_TIMEOUT,
)
from task1_object_detection.agent.tools import (
    ANALYZER_TOOLS,
    BOOSTER_TOOLS,
    TRAINER_TOOLS,
)

logger = logging.getLogger(__name__)


ANALYZER_SYSTEM_PROMPT = """\
You are an expert dataset analyst for object detection, specializing in grocery shelf detection \
for the NM i AI competition.

## Your Goal
Thoroughly analyze the COCO-format dataset to identify weaknesses that hurt model performance. \
Your analysis directly drives the augmentation and training strategy.

## Workflow
1. **Always start** by calling `analyze_class_distribution` to understand class imbalance.
2. Call `analyze_bbox_distribution` to understand object sizes and positions.
3. Call `analyze_image_stats` to understand per-image annotation density.
4. Call `identify_weak_categories` with an appropriate threshold to find problematic classes.
5. Optionally call `visualize_annotations` to spot annotation quality issues.
6. Optionally call `search_hf_models` if relevant pretrained models could help.

## Focus Areas
- **Class imbalance**: With 356 categories, many will be rare. Quantify the imbalance.
- **Annotation quality**: Look for anomalous bbox sizes (too small, too large, zero-area).
- **Bbox distribution**: Are objects mostly small? Centered? This affects augmentation choices.
- **Category grouping**: Can rare categories be grouped with similar ones?

## Output Format
Provide a structured analysis report:

DATASET ANALYSIS REPORT
=======================

CLASS DISTRIBUTION:
- Total categories: X, Total annotations: Y
- Rare classes (<10 annotations): N categories
- Key imbalance findings

BBOX ANALYSIS:
- Size distribution (small/medium/large)
- Aspect ratio insights
- Position distribution

WEAK POINTS:
- List specific weak categories and counts
- Suggested strategies for each

RECOMMENDATIONS:
1. Oversampling strategy for rare classes
2. Augmentation parameters to use
3. Model size recommendation
4. Any other insights

## Rules
- Be thorough but efficient — use all analysis tools.
- Provide actionable numbers, not vague statements.
- The scoring is 70% detection mAP@0.5 + 30% classification mAP@0.5.
"""

BOOSTER_SYSTEM_PROMPT = """\
You are a dataset preparation and augmentation specialist for object detection training. \
You prepare the dataset for optimal YOLOv8 training based on the analyzer's findings.

## Your Goal
Convert the COCO dataset to YOLO format and apply targeted augmentations to boost \
performance on weak categories.

## Workflow
1. Call `create_yolo_dataset` to convert COCO annotations to YOLO format with train/val split.
2. Based on the analysis, call `apply_oversampling` with appropriate factor for rare classes.
3. Call `generate_augmentation_config` with parameters tuned to the dataset characteristics.

## Augmentation Strategy Guidelines
- **Many small objects**: Use higher mosaic (1.0), moderate scale (0.3-0.5).
- **Heavy class imbalance**: Aggressive oversampling (5-10x) for very rare classes.
- **Varied object sizes**: Use scale augmentation (0.3-0.7).
- **Grocery products**: Light rotation (5-15 deg), moderate color aug, horizontal flip only.
- Products on shelves: Do NOT use vertical flip (flipud=0.0).

## Output Format
DATASET PREPARATION REPORT
===========================

YOLO CONVERSION:
- Train/val split: X/Y images
- Total classes: N

OVERSAMPLING:
- Rare classes boosted: N
- Duplication factor: X
- New training set size: Y images

AUGMENTATION CONFIG:
- Key parameters and rationale

READY FOR TRAINING:
- data.yaml path
- Recommended training command

## Rules
- Always create the YOLO dataset first before oversampling.
- Use the analysis findings to set appropriate thresholds.
- Save all configs for reproducibility.
"""

TRAINER_SYSTEM_PROMPT = """\
You are a YOLOv8 training specialist for object detection. You train, evaluate, and \
optimize models for the NM i AI grocery detection competition.

## Your Goal
Train the best possible YOLOv8 model. The competition score is \
70% detection mAP@0.5 + 30% classification mAP@0.5.

## Workflow
1. Call `train_yolo_model` with appropriate parameters based on the booster's report.
2. Call `evaluate_model` to get per-class metrics and identify weak categories.
3. Call `run_inference` to visually verify detection quality.
4. If time allows, consider a second training round with adjusted parameters.

## Model Selection Guidelines
- **yolov8n.pt**: Fast, good for iteration. Use for quick experiments.
- **yolov8s.pt**: Good balance for 248 images.
- **yolov8m.pt**: Best for final training if compute allows.
- **yolov8l.pt / yolov8x.pt**: Likely overkill for 248 images, risk of overfitting.

## Training Tips
- With 356 categories and 248 images, use strong regularization.
- Start with yolov8s or yolov8m, 50-100 epochs, patience=10.
- Image size 640 is standard; 1280 if objects are very small.
- Batch size: 16 (adjust based on memory).
- Use the augmentation config generated by the booster.

## Output Format
TRAINING REPORT
===============

MODEL:
- Architecture: yolov8X
- Epochs trained: N
- Best model path: ...

METRICS:
- mAP@0.5: X.XXX
- mAP@0.5:0.95: X.XXX
- Precision: X.XXX
- Recall: X.XXX
- Estimated competition score: X.XXX

WEAK CLASSES:
- Classes with 0 AP: [list]
- Classes with low AP: [list]

RECOMMENDATIONS:
- Next steps for improvement

## Rules
- Always evaluate after training.
- Report per-class metrics to identify systematic weaknesses.
- Save the best model weights.
"""


class ObjectDetectionAgent:
    """Orchestrates the analyzer → booster → trainer pipeline for object detection."""

    def __init__(self):
        self.analyzer_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0,
            max_retries=2,
            timeout=30.0,
        )
        self.booster_llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0,
            max_retries=2,
            timeout=30.0,
        )
        self.trainer_llm = ChatAnthropic(
            model="claude-opus-4-20250514",
            max_tokens=4096,
            temperature=0,
            max_retries=2,
            timeout=60.0,
        )
        self.fallback_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_retries=2,
            timeout=60,
        )

        # Analyzer: investigates the dataset
        self.analyzer = create_react_agent(
            model=self.analyzer_llm,
            tools=ANALYZER_TOOLS,
            prompt=ANALYZER_SYSTEM_PROMPT,
        )
        self.fallback_analyzer = create_react_agent(
            model=self.fallback_llm,
            tools=ANALYZER_TOOLS,
            prompt=ANALYZER_SYSTEM_PROMPT,
        )

        # Booster: prepares and augments the dataset
        self.booster = create_react_agent(
            model=self.booster_llm,
            tools=BOOSTER_TOOLS,
            prompt=BOOSTER_SYSTEM_PROMPT,
        )
        self.fallback_booster = create_react_agent(
            model=self.fallback_llm,
            tools=BOOSTER_TOOLS,
            prompt=BOOSTER_SYSTEM_PROMPT,
        )

        # Trainer: trains and evaluates models
        self.trainer = create_react_agent(
            model=self.trainer_llm,
            tools=TRAINER_TOOLS,
            prompt=TRAINER_SYSTEM_PROMPT,
        )
        self.fallback_trainer = create_react_agent(
            model=self.fallback_llm,
            tools=TRAINER_TOOLS,
            prompt=TRAINER_SYSTEM_PROMPT,
        )

    async def _run_with_fallback(self, primary, fallback, messages, config):
        """Run primary agent, fall back to Gemini on rate limit or server error."""
        try:
            return await primary.ainvoke(messages, config=config)
        except Exception as e:
            err = f"{type(e).__name__}: {e}".lower()
            if any(k in err for k in ["429", "rate", "503", "unavailable", "high demand", "overloaded", "servererror"]):
                logger.warning("Primary model unavailable (%s), falling back to Gemini", type(e).__name__)
                try:
                    return await fallback.ainvoke(messages, config=config)
                except Exception as e2:
                    logger.warning("Fallback also failed: %s", e2)
                    raise e
            raise

    def _extract_last_message(self, result: dict) -> str:
        """Extract text content from the last agent message."""
        last = result["messages"][-1]
        content = last.content
        if isinstance(content, list):
            return "\n".join(
                part.get("text", str(part)) if isinstance(part, dict) else str(part)
                for part in content
            )
        return content

    def _create_langfuse_handler(self) -> LangfuseCallbackHandler | None:
        """Create Langfuse callback handler if credentials are configured."""
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            return LangfuseCallbackHandler()
        return None

    async def analyze(self, task_description: str = "") -> str:
        """Run the analyzer to investigate the dataset."""
        config: dict = {}
        langfuse_handler = self._create_langfuse_handler()
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]
        config["recursion_limit"] = 25

        prompt = task_description or (
            "Analyze the grocery shelf detection dataset thoroughly. "
            "Run all analysis tools and provide a comprehensive report on class distribution, "
            "bbox statistics, image stats, and weak categories. "
            "Identify the key challenges and recommend strategies."
        )

        messages = {"messages": [HumanMessage(content=prompt)]}

        try:
            result = await asyncio.wait_for(
                self._run_with_fallback(self.analyzer, self.fallback_analyzer, messages, config),
                timeout=ANALYZER_TIMEOUT,
            )
            report = self._extract_last_message(result)
            logger.info("Analysis report:\n%s", report)
            return report
        except asyncio.TimeoutError:
            logger.warning("Analyzer timed out after %ds", ANALYZER_TIMEOUT)
            return "Analysis timed out. Check partial results in output/analysis/."
        except Exception as e:
            logger.exception("Analyzer failed: %s", e)
            return f"Analysis failed: {e}"

    async def boost(self, analysis_report: str = "") -> str:
        """Run the booster to prepare and augment the dataset."""
        config: dict = {}
        langfuse_handler = self._create_langfuse_handler()
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]
        config["recursion_limit"] = 25

        prompt = (
            "Prepare the dataset for YOLOv8 training. "
            "Convert COCO to YOLO format, apply oversampling for rare classes, "
            "and generate an optimal augmentation config.\n\n"
        )
        if analysis_report:
            prompt += f"## Analysis Report\n\n{analysis_report}"

        messages = {"messages": [HumanMessage(content=prompt)]}

        try:
            result = await asyncio.wait_for(
                self._run_with_fallback(self.booster, self.fallback_booster, messages, config),
                timeout=BOOSTER_TIMEOUT,
            )
            report = self._extract_last_message(result)
            logger.info("Boost report:\n%s", report)
            return report
        except asyncio.TimeoutError:
            logger.warning("Booster timed out after %ds", BOOSTER_TIMEOUT)
            return "Boost timed out. Check partial results in output/yolo_dataset/."
        except Exception as e:
            logger.exception("Booster failed: %s", e)
            return f"Boost failed: {e}"

    async def train(self, boost_report: str = "") -> str:
        """Run the trainer to train and evaluate a YOLOv8 model."""
        config: dict = {}
        langfuse_handler = self._create_langfuse_handler()
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]
        config["recursion_limit"] = 40

        prompt = (
            "Train a YOLOv8 model on the prepared dataset. "
            "Evaluate per-class performance and identify weak categories. "
            "Run inference on sample images to verify quality.\n\n"
        )
        if boost_report:
            prompt += f"## Dataset Preparation Report\n\n{boost_report}"

        messages = {"messages": [HumanMessage(content=prompt)]}

        try:
            result = await asyncio.wait_for(
                self._run_with_fallback(self.trainer, self.fallback_trainer, messages, config),
                timeout=TRAINER_TIMEOUT,
            )
            report = self._extract_last_message(result)
            logger.info("Training report:\n%s", report)
            return report
        except asyncio.TimeoutError:
            logger.warning("Trainer timed out after %ds", TRAINER_TIMEOUT)
            return "Training timed out. Check partial results in output/models/."
        except Exception as e:
            logger.exception("Trainer failed: %s", e)
            return f"Training failed: {e}"

    async def run_full_pipeline(self, task_description: str = "") -> dict[str, str]:
        """Run the full analyzer → booster → trainer pipeline."""
        start_time = time.time()
        reports = {}

        # Step 1: Analyze
        logger.info("=" * 60)
        logger.info("STEP 1: ANALYZING DATASET")
        logger.info("=" * 60)
        analysis_report = await self.analyze(task_description)
        reports["analysis"] = analysis_report
        logger.info("Analysis completed in %.1fs", time.time() - start_time)

        # Step 2: Boost
        logger.info("=" * 60)
        logger.info("STEP 2: BOOSTING DATASET")
        logger.info("=" * 60)
        boost_start = time.time()
        boost_report = await self.boost(analysis_report)
        reports["boost"] = boost_report
        logger.info("Boost completed in %.1fs", time.time() - boost_start)

        # Step 3: Train
        logger.info("=" * 60)
        logger.info("STEP 3: TRAINING MODEL")
        logger.info("=" * 60)
        train_start = time.time()
        train_report = await self.train(boost_report)
        reports["training"] = train_report
        logger.info("Training completed in %.1fs", time.time() - train_start)

        total = time.time() - start_time
        logger.info("=" * 60)
        logger.info("FULL PIPELINE COMPLETED in %.1fs", total)
        logger.info("=" * 60)

        reports["total_time_seconds"] = f"{total:.1f}"
        return reports
