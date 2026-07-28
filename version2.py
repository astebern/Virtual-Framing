"""Dynamic Interactive Virtual Framing with Gesture-Controlled Effects.

Run with: python3 version2.py [--camera 0]
Press C to capture the active ROI. Press Q to quit.
Captured frames are written to the captures/ directory.
"""

from __future__ import annotations

import argparse
import contextlib
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
    EFFECTS = ("None", "GaussianBlur", "B&W", "Canny", "Pixelate", "Invert")
    FINGER_TIPS = (4, 8, 12, 16, 20)
    FINGER_LAYERS = (
        ((80, 255, 140), 0.28),
        ((80, 80, 255), 0.34),
        ((255, 220, 70), 0.34),
        ((245, 245, 245), 0.26),
        ((0, 0, 0), 0.42),
    )

    def __init__(self, camera_id: int = 0, virtual_camera_device: str = "/dev/video20") -> None:
        self.camera_id = camera_id
        self.virtual_camera_device = virtual_camera_device
        self.effect_index = 0
        self.last_capture = 0.0
        self.last_effect_change = 0.0
        self.last_mirror_change = 0.0
        self.last_output_mode_change = 0.0
        self.last_box_mode_change = 0.0
        self.mirror_enabled = True
        self.clean_output = True
        self.five_finger_box = False
        self.was_right_pinching = False
        self.was_left_pinching = False
        self.was_index_touching = False
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

    def select_effect(self, index: int) -> None:
        if 0 <= index < len(self.EFFECTS):
            self.effect_index = index

    @staticmethod
    def pinch_threshold(frame_width: int, frame_height: int) -> float:
        # The threshold is normalized, so it works across camera resolutions.
        del frame_width, frame_height
        return 0.065

    @staticmethod
    def index_touch_threshold(frame_width: int, frame_height: int) -> float:
        return min(frame_width, frame_height) * 0.08

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

    def read_hands(self, frame: np.ndarray, mirrored: bool) -> list[HandState]:
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
            label = media_label if mirrored else ("Left" if media_label == "Right" else "Right")
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

    @classmethod
    def five_finger_polygon(cls, states: list[HandState]) -> Optional[np.ndarray]:
        points = [
            state.points[landmark_id]
            for state in states
            for landmark_id in cls.FINGER_TIPS
            if landmark_id in state.points
        ]
        if len(points) < 3:
            return None

        point_array = np.array(points, dtype=np.int32)
        return cv2.convexHull(point_array).reshape(-1, 2)

    @classmethod
    def five_finger_layers(
        cls,
        states: list[HandState],
    ) -> list[Tuple[np.ndarray, Tuple[int, int, int], float]]:
        right = next((state for state in states if state.label == "Right"), None)
        left = next((state for state in states if state.label == "Left"), None)
        if not right or not left:
            return []

        left_tips = np.array([left.points[landmark_id] for landmark_id in cls.FINGER_TIPS], dtype=np.float32)
        right_tips = np.array([right.points[landmark_id] for landmark_id in cls.FINGER_TIPS], dtype=np.float32)
        left_bounds = cls.finger_slice_bounds(left_tips)
        right_bounds = cls.finger_slice_bounds(right_tips)

        layers = []
        for index, (color, alpha) in enumerate(cls.FINGER_LAYERS):
            polygon = np.array(
                [
                    left_bounds[index],
                    right_bounds[index],
                    right_bounds[index + 1],
                    left_bounds[index + 1],
                ],
                dtype=np.int32,
            )
            layers.append((polygon, color, alpha))
        return layers

    @staticmethod
    def finger_slice_bounds(tips: np.ndarray) -> np.ndarray:
        bounds = [(tips[index] + tips[index + 1]) * 0.5 for index in range(len(tips) - 1)]
        first = tips[0] + (tips[0] - tips[1]) * 0.55
        last = tips[-1] + (tips[-1] - tips[-2]) * 0.55
        return np.array([first, *bounds, last], dtype=np.float32)

    @staticmethod
    def bounds_from_polygon(polygon: np.ndarray) -> Tuple[int, int, int, int]:
        x, y, width, height = cv2.boundingRect(polygon)
        return x, y, x + width - 1, y + height - 1

    @staticmethod
    def bounds_from_layers(
        layers: list[Tuple[np.ndarray, Tuple[int, int, int], float]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if not layers:
            return None

        points = np.vstack([polygon for polygon, _color, _alpha in layers])
        return VirtualFraming.bounds_from_polygon(points)

    @classmethod
    def index_fingers_touching(cls, states: list[HandState], frame_width: int, frame_height: int) -> bool:
        right = next((state for state in states if state.label == "Right"), None)
        left = next((state for state in states if state.label == "Left"), None)
        if not right or not left:
            return False

        distance = np.hypot(right.points[8][0] - left.points[8][0], right.points[8][1] - left.points[8][1])
        return bool(distance < cls.index_touch_threshold(frame_width, frame_height))

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
        if self.active_effect == "Canny":
            threshold = int(40 + (1.0 - intensity) * 160)
            edges = cv2.Canny(roi, threshold, min(255, threshold * 2))
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        if self.active_effect == "Pixelate":
            height, width = roi.shape[:2]
            block_size = int(3 + round(intensity * 22))
            small_width = max(1, width // block_size)
            small_height = max(1, height // block_size)
            small = cv2.resize(roi, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        inverted = cv2.bitwise_not(roi)
        return cv2.addWeighted(inverted, 0.35 + intensity * 0.65, roi, 0.65 - intensity * 0.65, 0)

    def draw_hand(self, frame: np.ndarray, state: HandState) -> None:
        for a, b in self.mp_hands.HAND_CONNECTIONS:
            cv2.line(frame, state.points[a], state.points[b], (255, 255, 255), 2, cv2.LINE_AA)
        for point in state.points.values():
            cv2.circle(frame, point, 3, (0, 0, 255), -1, cv2.LINE_AA)

    @staticmethod
    def draw_keypoint_labels(frame: np.ndarray, states: list[HandState]) -> None:
        for state in states:
            prefix = "R" if state.label == "Right" else "L"
            for landmark_id in (4, 8):
                x, y = state.points[landmark_id]
                cv2.putText(
                    frame,
                    f"{prefix}{landmark_id}",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 220, 80),
                    1,
                    cv2.LINE_AA,
                )

    @staticmethod
    def draw_five_finger_region(frame: np.ndarray, polygon: np.ndarray) -> None:
        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, polygon, (80, 255, 150), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)
        cv2.polylines(frame, [polygon], True, (80, 255, 150), 2, cv2.LINE_AA)

    @staticmethod
    def draw_five_finger_layers(
        frame: np.ndarray,
        layers: list[Tuple[np.ndarray, Tuple[int, int, int], float]],
    ) -> None:
        for polygon, color, alpha in layers:
            overlay = frame.copy()
            cv2.fillConvexPoly(overlay, polygon, color, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
            edge_color = tuple(min(255, channel + 35) for channel in color)
            cv2.polylines(frame, [polygon], True, edge_color, 1, cv2.LINE_AA)

    def draw_overlay(self, frame: np.ndarray, box_active: bool) -> None:
        height, width = frame.shape[:2]
        if box_active:
            cv2.putText(
                frame,
                "BOX ACTIVE",
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        mode_text = f"BOX MODE: {'5 FINGERS' if self.five_finger_box else '2 FINGERS'}"
        cv2.putText(
            frame,
            mode_text,
            (16, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 255, 150) if self.five_finger_box else (255, 220, 80),
            1,
            cv2.LINE_AA,
        )

        effect_name = "None" if self.five_finger_box else self.active_effect
        effect_text = f"ACTIVE EFFECT: {effect_name}"
        text_width = cv2.getTextSize(effect_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.putText(
            frame,
            effect_text,
            (max(16, width - text_width - 16), 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        mirror_text = f"MIRROR: {'ON' if self.mirror_enabled else 'OFF'}"
        mirror_text_width = cv2.getTextSize(mirror_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0][0]
        cv2.putText(
            frame,
            mirror_text,
            (max(16, width - mirror_text_width - 16), 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (180, 255, 180) if self.mirror_enabled else (190, 190, 190),
            1,
            cv2.LINE_AA,
        )

        output_text = f"OUTPUT: {'CLEAN' if self.clean_output else 'DEBUG'}"
        output_text_width = cv2.getTextSize(output_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0][0]
        cv2.putText(
            frame,
            output_text,
            (max(16, width - output_text_width - 16), 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (180, 255, 180) if self.clean_output else (255, 220, 80),
            1,
            cv2.LINE_AA,
        )

        instruction = "0-5: Effect | Fist: Box Mode | Right Pinch: Next | Left Pinch: Prev | Index Touch: Mirror | C: Capture | O: Output | Q: Quit"
        font_scale = 0.52
        instruction_width = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        if instruction_width > width - 24:
            font_scale = max(0.35, (width - 24) / instruction_width * font_scale)
        cv2.putText(
            frame,
            instruction,
            (12, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
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

        # Ambil properti kamera asli.
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(camera.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30

        try:
            with contextlib.ExitStack() as stack:
                cam = None
                try:
                    cam = stack.enter_context(pyvirtualcam.Camera(
                        width,
                        height,
                        fps,
                        fmt=pyvirtualcam.PixelFormat.RGB,
                        device=self.virtual_camera_device,
                    ))
                    print(f"Kamera Virtual Aktif: {cam.device}")
                except RuntimeError as exc:
                    print(f"Virtual camera nonaktif: {exc}")
                    print("Preview lokal tetap berjalan. Untuk output meeting, aktifkan v4l2loopback lebih dulu.")

                while True:
                    ok, frame = camera.read()
                    if not ok:
                        print("Could not read a frame from the webcam.")
                        break

                    if self.mirror_enabled:
                        frame = cv2.flip(frame, 1)
                    states = self.read_hands(frame, self.mirror_enabled)
                    box = self.box_from_hands(states)
                    now = time.monotonic()
                    right = next((state for state in states if state.label == "Right"), None)
                    left = next((state for state in states if state.label == "Left"), None)

                    index_touching = self.index_fingers_touching(states, frame.shape[1], frame.shape[0])
                    if index_touching and not self.was_index_touching and now - self.last_mirror_change > 0.8:
                        self.mirror_enabled = not self.mirror_enabled
                        self.last_mirror_change = now
                    self.was_index_touching = index_touching

                    right_pinch = bool(right and right.is_pinching)
                    left_pinch = bool(left and left.is_pinching)
                    if not self.five_finger_box and now - self.last_effect_change > 0.35:
                        if right_pinch and not self.was_right_pinching:
                            self.change_effect(1)
                            self.last_effect_change = now
                        elif left_pinch and not self.was_left_pinching:
                            self.change_effect(-1)
                            self.last_effect_change = now
                    self.was_right_pinching, self.was_left_pinching = right_pinch, left_pinch

                    fist = bool(states) and all(state.is_fist for state in states)
                    if fist and not self.was_fist and now - self.last_box_mode_change > 0.8:
                        self.five_finger_box = not self.five_finger_box
                        self.last_box_mode_change = now
                    self.was_fist = fist

                    processed_frame = frame.copy()
                    active_box = None
                    five_finger_layers = []
                    five_finger_polygon = self.five_finger_polygon(states) if self.five_finger_box else None
                    if self.five_finger_box and five_finger_polygon is not None:
                        five_finger_layers = self.five_finger_layers(states)
                        active_box = self.bounds_from_layers(five_finger_layers)
                        if active_box is None:
                            active_box = self.bounds_from_polygon(five_finger_polygon)
                    elif box:
                        x1, y1, x2, y2 = box
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                        if x2 > x1 and y2 > y1:
                            active_box = (x1, y1, x2, y2)
                            pinch_distances = [state.pinch_distance for state in states]
                            average_distance = float(np.mean(pinch_distances)) if pinch_distances else 0.0
                            processed_frame[y1:y2, x1:x2] = self.apply_effect(
                                processed_frame[y1:y2, x1:x2],
                                average_distance,
                            )

                    effect_frame = processed_frame.copy()
                    if five_finger_layers:
                        self.draw_five_finger_layers(effect_frame, five_finger_layers)
                    elif five_finger_polygon is not None:
                        self.draw_five_finger_region(effect_frame, five_finger_polygon)

                    clean_frame = effect_frame.copy()
                    for state in states:
                        self.draw_hand(clean_frame, state)

                    debug_frame = clean_frame.copy()
                    if active_box and not self.five_finger_box:
                        x1, y1, x2, y2 = active_box
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

                    self.draw_keypoint_labels(debug_frame, states)
                    self.draw_overlay(debug_frame, active_box is not None)

                    output_frame = clean_frame if self.clean_output else debug_frame
                    if cam:
                        # Konversi warna ke format RGB yang dibaca aplikasi meeting.
                        frame_rgb_out = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

                        # Kirim hasil ke kamera virtual.
                        cam.send(frame_rgb_out)
                        cam.sleep_until_next_frame()

                    # Tampilkan juga di layar lokal untuk pantauan.
                    cv2.imshow("Dynamic Virtual Framing", output_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("c"), ord("C")) and active_box and now - self.last_capture > 0.3:
                        self.capture(effect_frame, active_box)
                        self.last_capture = now
                    elif key in (ord("o"), ord("O")) and now - self.last_output_mode_change > 0.3:
                        self.clean_output = not self.clean_output
                        self.last_output_mode_change = now
                    elif ord("0") <= key <= ord("9") and not self.five_finger_box:
                        self.select_effect(key - ord("0"))
                    if key in (ord("q"), ord("Q")):
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
