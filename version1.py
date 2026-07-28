"""Dynamic Interactive Virtual Framing with Gesture-Controlled Effects.

Run with: python virtual_framing.py [--camera 0]
Press Q to quit. Captured frames are written to the captures/ directory.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam


LandmarkPoint = Tuple[int, int]


@dataclass
class HandState:
    label: str
    points: Dict[int, LandmarkPoint]
    pinch_distance: float
    is_pinching: bool
    is_fist: bool


class VirtualFraming:
    EFFECTS = ("None", "GaussianBlur", "B&W", "Canny")

    def __init__(self, camera_id: int = 0, virtual_camera_device: str = "/dev/video20") -> None:
        self.camera_id = camera_id
        self.virtual_camera_device = virtual_camera_device
        self.effect_index = 0
        self.last_capture = 0.0
        self.last_effect_change = 0.0
        self.was_right_pinching = False
        self.was_left_pinching = False
        self.was_fist = False
        self.capture_dir = "captures"

        os.makedirs(self.capture_dir, exist_ok=True)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
        )

    @property
    def active_effect(self) -> str:
        return self.EFFECTS[self.effect_index]

    def change_effect(self, step: int) -> None:
        self.effect_index = (self.effect_index + step) % len(self.EFFECTS)

    @staticmethod
    def pinch_threshold(frame_width: int, frame_height: int) -> float:
        # The threshold is normalized, so it works across camera resolutions.
        del frame_width, frame_height
        return 0.065

    @staticmethod
    def is_fist(landmarks) -> bool:
        # For each finger, its tip should be closer to the wrist than its PIP.
        wrist = landmarks[0]
        tip_pip_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
        folded = 0
        for tip, pip in tip_pip_pairs:
            tip_distance = np.hypot(landmarks[tip].x - wrist.x, landmarks[tip].y - wrist.y)
            pip_distance = np.hypot(landmarks[pip].x - wrist.x, landmarks[pip].y - wrist.y)
            folded += tip_distance < pip_distance
        # Thumb is checked against the index MCP as a practical fist heuristic.
        thumb_folded = np.hypot(landmarks[4].x - landmarks[0].x, landmarks[4].y - landmarks[0].y) < np.hypot(
            landmarks[5].x - landmarks[0].x, landmarks[5].y - landmarks[0].y
        )
        return folded >= 3 and thumb_folded

    def read_hands(self, frame: np.ndarray) -> list[HandState]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        states: list[HandState] = []

        if not result.multi_hand_landmarks:
            return states

        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            # The input frame is mirrored before processing, matching MediaPipe's
            # selfie-image convention, so its handedness label can be used as-is.
            media_label = handedness.classification[0].label
            label = media_label
            points = {
                index: (int(point.x * width), int(point.y * height))
                for index, point in enumerate(hand_landmarks.landmark)
            }
            distance = float(np.hypot(
                hand_landmarks.landmark[4].x - hand_landmarks.landmark[8].x,
                hand_landmarks.landmark[4].y - hand_landmarks.landmark[8].y,
            ))
            states.append(HandState(
                label=label,
                points=points,
                pinch_distance=distance,
                is_pinching=distance < self.pinch_threshold(width, height),
                is_fist=self.is_fist(hand_landmarks.landmark),
            ))
        return states

    @staticmethod
    def box_from_hands(states: list[HandState]) -> Optional[Tuple[int, int, int, int]]:
        required = {}
        for state in states:
            required[f"{'R' if state.label == 'Right' else 'L'}8"] = state.points[8]
            required[f"{'R' if state.label == 'Right' else 'L'}4"] = state.points[4]
        if len(required) != 4:
            return None
        x_values = [point[0] for point in required.values()]
        y_values = [point[1] for point in required.values()]
        return min(x_values), min(y_values), max(x_values), max(y_values)

    def apply_effect(self, roi: np.ndarray, average_pinch_distance: float) -> np.ndarray:
        # Normalized distance maps to stronger effects when fingers are close.
        intensity = float(np.clip(1.0 - (average_pinch_distance - 0.035) / 0.25, 0.0, 1.0))
        if self.active_effect == "None":
            return roi.copy()
        if self.active_effect == "GaussianBlur":
            kernel = int(3 + round(intensity * 20))
            kernel = kernel if kernel % 2 else kernel + 1
            return cv2.GaussianBlur(roi, (kernel, kernel), 0)
        if self.active_effect == "B&W":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        threshold = int(40 + (1.0 - intensity) * 160)
        edges = cv2.Canny(roi, threshold, min(255, threshold * 2))
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def draw_hand(self, frame: np.ndarray, state: HandState, landmarks) -> None:
        self.mp_drawing.draw_landmarks(
            frame, landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
        )

    def capture(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = box
        crop = frame[max(0, y1):y2 + 1, max(0, x1):x2 + 1]
        if crop.size:
            filename = os.path.join(self.capture_dir, f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(filename, crop)
            print(f"Captured: {filename}")

    def run(self) -> None:
            camera = cv2.VideoCapture(self.camera_id)
            if not camera.isOpened():
                raise RuntimeError(f"Could not open webcam {self.camera_id}")

            # Ambil properti kamera asli
            width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(camera.get(cv2.CAP_PROP_FPS))
            if fps == 0: 
                fps = 30

            try:
                # Buka Virtual Camera
                with pyvirtualcam.Camera(
                    width,
                    height,
                    fps,
                    fmt=pyvirtualcam.PixelFormat.RGB,
                    device=self.virtual_camera_device,
                ) as cam:
                    print(f"Kamera Virtual Aktif: {cam.device}")
                    
                    while True:
                        ok, frame = camera.read()
                        if not ok:
                            print("Could not read a frame from the webcam.")
                            break
                        
                        # Mirror the camera horizontally for a natural selfie view.
                        frame = cv2.flip(frame, 1)
                        states = self.read_hands(frame)
                        box = self.box_from_hands(states)
                        now = time.monotonic()
                        right = next((state for state in states if state.label == "Right"), None)
                        left = next((state for state in states if state.label == "Left"), None)

                        right_pinch = bool(right and right.is_pinching)
                        left_pinch = bool(left and left.is_pinching)
                        if now - self.last_effect_change > 0.35:
                            if right_pinch and not self.was_right_pinching:
                                self.change_effect(1)
                                self.last_effect_change = now
                            elif left_pinch and not self.was_left_pinching:
                                self.change_effect(-1)
                                self.last_effect_change = now
                        self.was_right_pinching, self.was_left_pinching = right_pinch, left_pinch

                        fist = bool(states) and all(state.is_fist for state in states)
                        if fist and not self.was_fist and box and now - self.last_capture > 1.0:
                            self.capture(frame, box)
                            self.last_capture = now
                        self.was_fist = fist

                        if box:
                            x1, y1, x2, y2 = box
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                            if x2 > x1 and y2 > y1:
                                pinch_distances = [state.pinch_distance for state in states]
                                average_distance = float(np.mean(pinch_distances)) if pinch_distances else 0.0
                                frame[y1:y2, x1:x2] = self.apply_effect(frame[y1:y2, x1:x2], average_distance)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

                        # Draw tracking visuals after the ROI effect so they remain crisp.
                        if self.hands:  # landmarks are redrawn by processing a second lightweight result is avoided below
                            pass
                        # Re-process only for drawing metadata retained by MediaPipe is not exposed;
                        # use the pixel points to render the required skeleton/landmarks from states.
                        for state in states:
                            for a, b in self.mp_hands.HAND_CONNECTIONS:
                                cv2.line(frame, state.points[a], state.points[b], (255, 255, 255), 2, cv2.LINE_AA)
                            for point in state.points.values():
                                cv2.circle(frame, point, 3, (0, 0, 255), -1, cv2.LINE_AA)
                        
                        # --- BAGIAN VIRTUAL CAM ---
                        # Konversi warna ke format RGB yang dibaca aplikasi meet
                        frame_rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Kirim hasil ke kamera virtual
                        cam.send(frame_rgb_out)
                        cam.sleep_until_next_frame()

                        # Tampilkan juga di layar lokal untuk pantauan
                        cv2.imshow("Dynamic Virtual Framing", frame)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                            break
            finally:
                camera.release()
                self.hands.close()
                cv2.destroyAllWindows()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="webcam device ID (default: 0)")
    parser.add_argument(
        "--virtual-camera",
        default="/dev/video20",
        help="virtual camera device path (default: /dev/video20)",
    )
    args = parser.parse_args()
    VirtualFraming(args.camera, args.virtual_camera).run()


if __name__ == "__main__":
    main()
