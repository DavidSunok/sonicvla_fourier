"""Lightweight Fourier FDH-6 hand driver for VLA inference.

Handles device discovery and joint command/state via the dexhandpy SDK.
No retargeting or landmark dependencies — just direct 6D joint control.
"""

from __future__ import annotations

import threading
import time

import numpy as np

FOURIER_NUM_MOTORS = 6

# Hardware joint order: [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
# SDK set_pos order:    [index, middle, ring, pinky, thumb_pitch, thumb_yaw]
HARDWARE_TO_SDK_IDX = [2, 3, 4, 5, 1, 0]

# Joint ranges (radians, hardware order).
# thumb_pitch is reversed: positive rad = open.
JOINT_RAD_RANGES = [
    (-1.676, 0.0),   # thumb_yaw
    (1.159, 0.0),    # thumb_pitch (reversed)
    (-1.602, 0.0),   # index
    (-1.603, 0.0),   # middle
    (-1.602, 0.0),   # ring
    (-1.602, 0.0),   # pinky
]


def _rad_to_normalized(q_rad: float, hardware_idx: int) -> float:
    min_val, max_val = JOINT_RAD_RANGES[hardware_idx]
    return float(np.clip((q_rad - min_val) / (max_val - min_val), 0.0, 1.0))


def _normalized_to_rad(q_normalized: float, hardware_idx: int) -> float:
    min_val, max_val = JOINT_RAD_RANGES[hardware_idx]
    q = float(np.clip(q_normalized, 0.0, 1.0))
    return float(q * (max_val - min_val) + min_val)


class FourierInferenceHand:
    """Minimal Fourier FDH-6 driver for inference — send 6D, read 6D."""

    _INIT_RETRY_ATTEMPTS = 5
    _INIT_RETRY_SLEEP_S = 1.0
    _REDISCOVER_INTERVAL_S = 2.0

    def __init__(self, simulation_mode: bool = False, action_scale: float = 1.0) -> None:
        self.simulation_mode = simulation_mode
        self.action_scale = action_scale
        self._prev_left = np.zeros(FOURIER_NUM_MOTORS, dtype=np.float32)
        self._prev_left[0] = -1.676  # thumb_yaw constant
        self._prev_right = np.zeros(FOURIER_NUM_MOTORS, dtype=np.float32)
        self._prev_right[0] = -1.676  # thumb_yaw constant
        self.left_ip: str | None = None
        self.right_ip: str | None = None
        self.fdh = None
        self._sdk_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._rediscover_thread: threading.Thread | None = None

        if not simulation_mode:
            self._init_sdk()
            self._start_rediscover_thread()
            print(
                f"[FourierInferenceHand] Ready: "
                f"left={self.left_ip}, right={self.right_ip}, "
                f"sim={simulation_mode}, action_scale={action_scale}"
            )
        else:
            print("[FourierInferenceHand] Simulation mode — no hardware")

    # -- SDK init / discovery ------------------------------------------------

    def _init_sdk(self) -> None:
        try:
            import dexhandpy.fdexhand as fdh
        except ImportError:
            raise ImportError(
                "dexhandpy is required. Install it in .venv_inference: "
                "see install_scripts/install_inference.sh"
            )

        for attempt in range(1, self._INIT_RETRY_ATTEMPTS + 1):
            self.fdh = fdh.DexHand()
            ret = self.fdh.init()
            if ret != fdh.Ret.SUCCESS:
                print(
                    f"[FourierInferenceHand] SDK init failed ({ret}), "
                    f"attempt {attempt}/{self._INIT_RETRY_ATTEMPTS}"
                )
                self.fdh = None
                if attempt < self._INIT_RETRY_ATTEMPTS:
                    time.sleep(self._INIT_RETRY_SLEEP_S)
                continue

            self._discover_devices(log=True)
            if self.left_ip and self.right_ip:
                break

            if attempt < self._INIT_RETRY_ATTEMPTS:
                print("[FourierInferenceHand] Incomplete discovery, retrying...")
                time.sleep(self._INIT_RETRY_SLEEP_S)

        if self.fdh is None:
            print("[FourierInferenceHand] SDK init failed; will retry in background")
        else:
            if not self.left_ip:
                print("[FourierInferenceHand] WARNING: left hand (FDH-6L) not found")
            if not self.right_ip:
                print("[FourierInferenceHand] WARNING: right hand (FDH-6R) not found")

    def _discover_devices(self, log: bool = False) -> None:
        if self.fdh is None:
            return
        with self._sdk_lock:
            try:
                ip_list = self.fdh.get_ip_list()
            except Exception as exc:
                if log:
                    print(f"[FourierInferenceHand] Discovery failed: {exc}")
                return

            prev_left, prev_right = self.left_ip, self.right_ip
            for ip in ip_list:
                try:
                    hand_type = self.fdh.get_type(ip)
                except Exception:
                    continue
                if "L" in hand_type:
                    self.left_ip = ip
                elif "R" in hand_type:
                    self.right_ip = ip

        if log and (self.left_ip != prev_left or self.right_ip != prev_right):
            print(
                f"[FourierInferenceHand] Devices: "
                f"left={self.left_ip}, right={self.right_ip}"
            )

    def _start_rediscover_thread(self) -> None:
        self._rediscover_thread = threading.Thread(
            target=self._rediscover_loop,
            name="fourier-inference-rediscover",
            daemon=True,
        )
        self._rediscover_thread.start()

    def _rediscover_loop(self) -> None:
        while not self._stop_event.wait(self._REDISCOVER_INTERVAL_S):
            if self.fdh is None:
                self._init_sdk()
                continue
            if self.left_ip is None or self.right_ip is None:
                self._discover_devices(log=True)

    # -- Public API ----------------------------------------------------------

    def send_joints(self, left_q: np.ndarray, right_q: np.ndarray) -> None:
        """Send 6D joint targets (hardware order, radians) to both hands.

        Uses action_scale to interpolate between previous and target position,
        reducing sudden movements.  scale=1.0 → full command, scale=0.1 → very slow.
        """
        if self.simulation_mode:
            return
        if self.fdh is None:
            return

        left_q = self._prev_left + self.action_scale * (left_q - self._prev_left)
        right_q = self._prev_right + self.action_scale * (right_q - self._prev_right)
        # thumb_yaw is always fixed at -1.676, never interpolated
        left_q[0] = -1.676
        right_q[0] = -1.676
        self._prev_left = left_q.copy()
        self._prev_right = right_q.copy()

        for ip, q in [(self.left_ip, left_q), (self.right_ip, right_q)]:
            if not ip:
                continue
            try:
                with self._sdk_lock:
                    sdk_pos = [
                        _rad_to_normalized(float(q[HARDWARE_TO_SDK_IDX[i]]), HARDWARE_TO_SDK_IDX[i])
                        for i in range(FOURIER_NUM_MOTORS)
                    ]
                    self.fdh.set_pos(ip, sdk_pos)
            except Exception as exc:
                print(f"[FourierInferenceHand] Send error ({ip}): {exc}")

    def read_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Read current joint positions (hardware order, radians).

        Returns (left_q, right_q), each shape (6,).
        """
        if self.simulation_mode or self.fdh is None:
            return (
                np.zeros(FOURIER_NUM_MOTORS, dtype=np.float32),
                np.zeros(FOURIER_NUM_MOTORS, dtype=np.float32),
            )

        left_state = np.zeros(FOURIER_NUM_MOTORS, dtype=np.float32)
        right_state = np.zeros(FOURIER_NUM_MOTORS, dtype=np.float32)

        for ip, out in [(self.left_ip, left_state), (self.right_ip, right_state)]:
            if not ip:
                continue
            try:
                with self._sdk_lock:
                    pos = self.fdh.get_pos(ip)
                if pos is None:
                    continue
                arr = np.asarray(pos, dtype=np.float32).reshape(-1)
                if arr.size < FOURIER_NUM_MOTORS:
                    continue
                for sdk_idx, hw_idx in enumerate(HARDWARE_TO_SDK_IDX):
                    out[hw_idx] = _normalized_to_rad(arr[sdk_idx], hw_idx)
            except Exception as exc:
                print(f"[FourierInferenceHand] Read error ({ip}): {exc}")

        return left_state, right_state

    def close(self) -> None:
        self._stop_event.set()
