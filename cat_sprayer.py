#!/usr/bin/env python3
"""Cat detector for Raspberry Pi 5, Picamera2, and LiteRT."""

import argparse
from contextlib import ExitStack
import logging
from pathlib import Path
import time

PROJECT_DIR = Path(__file__).resolve().parent
MODEL = PROJECT_DIR / "model" / "detect.tflite"
LOG = PROJECT_DIR / "detections.log"
CHIP = "/dev/gpiochip0"
RELAY = 17
CAT_ID = 16
CONF = 0.50
FRAMES = 3
SPRAY = 5
COOLDOWN = 30

log = logging.getLogger("catdet")


def open_relay(stack):
    # Only called in water mode. Dry-run never imports gpiod or requests a line.
    import gpiod
    from gpiod.line import Direction, Value

    request = stack.enter_context(gpiod.request_lines(
        CHIP,
        consumer="catdet",
        config={RELAY: gpiod.LineSettings(
            direction=Direction.OUTPUT, output_value=Value.INACTIVE,
        )},
    ))
    stack.callback(request.set_value, RELAY, Value.INACTIVE)

    def spray(seconds):
        log.info(">>> SPRAYING %ss <<<", seconds)
        try:
            request.set_value(RELAY, Value.ACTIVE)
            time.sleep(seconds)
        finally:
            request.set_value(RELAY, Value.INACTIVE)
        log.info("Spray done")

    return spray


def dry_run_spray(seconds):
    log.info("[DRY RUN] Would spray for %ss", seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Detect without accessing GPIO (default)")
    mode.add_argument("--enable-water", action="store_true", help="Enable the physical sprinkler output")
    args = parser.parse_args()

    import cv2
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
    from picamera2 import Picamera2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
    )
    log.info("Mode: %s", "WATER ENABLED" if args.enable_water else "DRY RUN (GPIO disabled)")
    log.info("Loading model...")
    interpreter = Interpreter(model_path=str(MODEL))
    interpreter.allocate_tensors()
    model_input = interpreter.get_input_details()[0]
    outputs = interpreter.get_output_details()
    height, width = model_input["shape"][1:3]
    log.info("Model ready (%sx%s)", width, height)

    try:
        with ExitStack() as stack:
            log.info("Starting camera...")
            camera = stack.enter_context(Picamera2())
            camera.configure(camera.create_still_configuration(
                main={"size": (640, 480), "format": "RGB888"},
            ))
            camera.start()
            time.sleep(2)
            log.info("Camera ready")
            spray = open_relay(stack) if args.enable_water else dry_run_spray

            count = 0
            last_spray = -float("inf")
            last_report = -float("inf")
            while True:
                frame = camera.capture_array()
                resized = cv2.resize(frame, (width, height))
                interpreter.set_tensor(model_input["index"], np.expand_dims(resized, 0))
                interpreter.invoke()
                classes = interpreter.get_tensor(outputs[1]["index"])[0]
                scores = interpreter.get_tensor(outputs[2]["index"])[0]
                now = time.monotonic()
                if now - last_report >= 3:
                    top = int(np.argmax(scores))
                    log.info("Top detection: class=%s score=%.2f", int(classes[top]), scores[top])
                    last_report = now
                cat_scores = [
                    score for label, score in zip(classes, scores)
                    if int(label) == CAT_ID and score > CONF
                ]
                if cat_scores:
                    count += 1
                    log.info("Cat! conf=%.0f%% (%s/%s)", max(cat_scores) * 100, count, FRAMES)
                else:
                    count = max(0, count - 1)
                if count >= FRAMES and now - last_spray > COOLDOWN:
                    spray(SPRAY)
                    last_spray = time.monotonic()
                    count = 0
    except KeyboardInterrupt:
        log.info("Shutting down")


if __name__ == "__main__":
    main()
