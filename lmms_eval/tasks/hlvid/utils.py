"""HLVid task utilities for lmms_eval.

HLVid (bfshi/HLVid) is a video QA benchmark with 268 multiple-choice
questions. The question text includes options (A/B/C/D) inline and a
"Please answer directly with the letter" instruction.

Schema (from probe):
  question_id, category, video_path, question, answer
  answer is a single letter (A/B/C/D)
"""

import os
import re
import sys

from loguru import logger as eval_logger

HLVID_DATA_DIR = "/datasets/chanwutk/stove/hlvid/videos"


def hlvid_doc_to_visual(doc):
    """Return the video path for this document."""
    video_path = doc.get("video_path", "")

    if not video_path:
        eval_logger.error(f"Empty video_path in doc. Keys: {list(doc.keys())}")
        sys.exit(1)

    if not os.path.isabs(video_path):
        video_path = os.path.join(HLVID_DATA_DIR, video_path)

    if not os.path.exists(video_path):
        eval_logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    return [video_path]


def hlvid_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Format the MCQ prompt.

    HLVid questions already include options and instruction inline, e.g.:
      "What number is displayed...?\nA. 4608\nB. 4606\n...\nPlease answer directly..."

    We add a post_prompt if configured.
    """
    question = doc.get("question", "")

    post_prompt = ""
    if lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return question + post_prompt


def hlvid_process_results(doc, results):
    """Extract predicted answer and compare to ground truth."""
    pred_raw = results[0].strip()
    match = re.search(r"[ABCD]", pred_raw.upper())
    pred = match.group(0) if match else pred_raw[0].upper() if pred_raw else "X"

    gt = doc.get("answer", "").strip().upper()

    return {
        "hlvid_accuracy": {
            "pred": pred,
            "gt": gt,
            "correct": int(pred == gt),
            "category": doc.get("category", ""),
            "question_id": doc.get("question_id", ""),
        }
    }


def hlvid_aggregate_results(results):
    """Compute overall accuracy."""
    correct = sum(r["correct"] for r in results)
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    eval_logger.info(f"HLVid Accuracy: {accuracy * 100:.1f}% ({correct}/{total})")

    return accuracy * 100
