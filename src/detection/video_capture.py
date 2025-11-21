import math
import os
import pathlib
import queue
import time
from threading import Thread
from typing import List, Union

import cv2
import numpy as np

from elasticsearch_logging import get_logger  # noqa
logger = get_logger(__name__)


class VideoCapture:
    def __init__(
        self,
        stream_address: Union[str, pathlib.Path],
        get_latest_frame: bool = True,
        wait_seconds: float = 5.0,
        skip_rate_calc_method: str = "static",
        static_skip_rate: int = 1,
    ):
        """Creates video capture object that runs in parallel to main thread.

        Args:
            stream_address (str/pathlib.Path): Ip address or path to video to be streamed
            get_latest_frame (bool): When true, always selects the latest and current frame from buffer.
                When false, video may lag behind due to inference time latency. Recommended True for IP cameras and
                False for saved video. Default True.
            wait_seconds (float): Number of seconds to wait for a new frame before quitting
            skip_rate_calc_method (str): Method to use for calculating the streaming skip rate. "static" or "dynamic". Default is "static".
            static_skip_rate (int): The static skip rate to use if the method is 'static'. Default is 1 (which means NO skipping).
        """
        # open cv requires video or stream to be string
        if type(stream_address) == pathlib.PosixPath:
            stream_address = str(stream_address)

        self.cap = cv2.VideoCapture(stream_address)
        if not self.cap.isOpened():
            logger.info("Unable to initialize stream connection... exiting")
            self.exit_streamer()

        self.frame_buffer: queue.Queue = queue.Queue()
        self.get_latest_frame = get_latest_frame
        self.wait_seconds = wait_seconds

        self.stream_finished = False
        self.stop_thread = False

        self.frame_times: List[float] = []
        self.window_size = 100

        self.frame_count = 0

        self.skip_rate_calc_method = skip_rate_calc_method
        self.static_skip_rate = static_skip_rate

        # create a thread to read frames
        # this is required because we want to enable the ability
        # to get the latest frame for a buffer
        self._t = Thread(target=self._reader)
        self._t.daemon = True
        self._t.start()

    def read(self):
        """
        read frame from frame buffer
        """
        avg_fps = np.round(self.get_average_fps(), 2).item()
        os.environ["AVG_FPS"] = str(avg_fps)

        inf_fps = int(float(os.getenv("DET_INF_FPS", avg_fps)))
        # Calculate skip rate
        skip_rate = self.calculate_skip_rate(
            avg_fps,
            inf_fps,
            method=self.skip_rate_calc_method,
            static_skip_rate=self.static_skip_rate,
        )

        if not self.stream_finished:
            try:
                for _ in range(int(skip_rate)):
                    # Wait max 1 second
                    _new_frame = self.frame_buffer.get(True, self.wait_seconds)
                    self.frame_times.append(time.time())
                    if len(self.frame_times) > self.window_size:
                        self.frame_times.pop(0)
                    self.frame_count += 1
                return _new_frame
            except queue.Empty:
                logger.warning("Did not get a new frame anymore")
                logger.warning(
                    f"Status of the stream_finished now: {self.stream_finished}"
                )

                if self.stream_finished:
                    return None

        return None

    def get_fps(self):
        """
        Calculate the current frames per second (FPS).
        """
        if len(self.frame_times) < 2:
            return 0.0
        return 1 / (self.frame_times[-1] - self.frame_times[-2])

    def get_average_fps(self):
        """
        Calculate the average FPS over the window size.
        """
        if len(self.frame_times) < 2:
            return 0.0
        window_start = max(0, len(self.frame_times) - self.window_size)
        return (len(self.frame_times) - window_start) / (
            self.frame_times[-1] - self.frame_times[window_start]
        )

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        logger.info("Starting video capture thread.")
        while not self.stop_thread:
            ret, self.frame = self.cap.read()

            if self.get_latest_frame:
                if not self.frame_buffer.empty():
                    try:
                        self.frame_buffer.get_nowait()  # discard previous (unprocessed) frame
                    except queue.Empty:
                        pass

            if not ret and self.frame_buffer.empty() and not self.stream_finished:
                logger.warning(
                    "Video reading complete and buffer is empty. Streaming will stop soon."
                )
                self.stream_finished = True

            self.frame_buffer.put(self.frame)

    def exit_streamer(self):
        logger.info("Exiting the streamer.")
        self.stop_thread = True
        logger.info("Stopping streamer thread")
        self._t.join()
        logger.info("Finished stopping streamer thread")
        if self.cap.isOpened():
            logger.info("VideoCapture was NOT yet closed, closing it now")
            self.cap.release()
            logger.info("Finished releasing the videocapture")
        logger.info("Exiting the streamer.")

    def calculate_skip_rate(
        self, avg_fps: float, inf_fps: float, method: str, static_skip_rate: int
    ) -> int:
        """
        Calculate the skip rate based on the given method.

        Parameters:
        avg_fps (float): The average frames per second.
        inf_fps (float): The inference frames per second.
        method (str): The method to use for calculating the skip rate. Can be 'static' or 'dynamic'.
        static_skip_rate (int): The static skip rate to use if the method is 'static'.

        Returns:
        int: The calculated skip rate.
        """
        if method == "static":
            return static_skip_rate
        return math.ceil(avg_fps / inf_fps)


if __name__ == "__main__":
    stream_address = pathlib.Path("../../test/assets/videos/friona1.mp4")
    streamer = VideoCapture(stream_address, get_latest_frame=False)
    frame_count = 0
    while True:
        frame = streamer.read()
        if frame is None:
            logger.warning("Either the stream is complete or camera disconnected.")
            streamer.exit_streamer()
            break
        frame_count += 1
        logger.info(
            f"Frame count: {frame_count}, FPS: {streamer.get_fps():.2f}, Average FPS: {streamer.get_average_fps():.2f}"
        )
        time.sleep(1)
