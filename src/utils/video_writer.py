import logging
import os
from pathlib import Path
from typing import Any, Tuple

import cv2

logger = logging.getLogger(__name__)


class VideoWriter:
    def __init__(
        self,
        video_reader: Any,
        video_save_path: Any,
        video_size: Tuple,
        video_speed: float = 1.0,
    ):
        """Creates a video writer object to save open cv videos
        Args:
            video_reader (cv2.VideoCapture): OpenCV video capture object that is to be recorded.
            video_save_path (str/Path): Path to output video. If None, video is not saved.
            video_size (Tuple): Output video size (width, height).
            video_speed (float): Target video speed. Default 1.0, meaning 1.0 * original video speed.
        """

        if video_save_path is None:
            logger.info("No video save path provided, video will not be saved.")
            return None

        self._get_video_path(video_save_path)

        self.writer = cv2.VideoWriter(
            str(self.video_save_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            video_reader.get(5) * video_speed,
            video_size,
        )

    def _get_video_path(self, video_save_path: Path) -> None:
        """Modifies video save path for inference or original
        Args:
            video_save_path (Path): Output video path
        """

        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

        self.__video_save_path = video_save_path

    @property
    def video_save_path(self):
        return self.__video_save_path

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
