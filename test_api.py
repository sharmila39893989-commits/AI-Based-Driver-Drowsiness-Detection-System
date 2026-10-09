from __future__ import annotations

import asyncio
import time
import unittest
from pathlib import Path

import cv2
import httpx
import numpy as np
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from backend.app import app


class DetectionApiTests(unittest.TestCase):
    def request(self, method: str, path: str, **kwargs):
        async def send_request():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send_request())

    def test_health_and_built_page_routes(self) -> None:
        self.assertEqual(self.request("GET", "/health").json(), {"status": "ok"})
        for path in ("/", "/live", "/analytics", "/alerts", "/history", "/about"):
            with self.subTest(path=path):
                response = self.request("GET", path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.headers["content-type"])

    def test_empty_upload_returns_actionable_error(self) -> None:
        response = self.request(
            "POST",
            "/detect",
            files={"file": ("empty.png", b"", "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["success"])
        self.assertIn("empty", response.json()["error"].lower())

    def test_valid_image_without_a_face_returns_no_face_result(self) -> None:
        encoded, image_data = cv2.imencode(
            ".png",
            np.zeros((96, 96, 3), dtype=np.uint8),
        )
        self.assertTrue(encoded)
        response = self.request(
            "POST",
            "/detect",
            files={"file": ("blank.png", image_data.tobytes(), "image/png")},
        )

        result = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(result["success"])
        self.assertFalse(result["face_detected"])
        self.assertIsNone(result["safety_score"])

    def test_yawn_and_drowsiness_detection_receive_the_uploaded_frame(self) -> None:
        encoded, image_data = cv2.imencode(
            ".png",
            np.zeros((96, 96, 3), dtype=np.uint8),
        )
        self.assertTrue(encoded)
        face_image = np.full((48, 32, 3), 255, dtype=np.uint8)

        with (
            patch(
                "backend.app.detect_driver_features",
                return_value=(
                    np.zeros((96, 96, 3), dtype=np.uint8),
                    "Head Normal",
                    False,
                    [],
                    face_image,
                ),
            ),
            patch(
                "backend.app.detect_yawn",
                return_value=("No Yawn", 0.9),
            ) as detect_yawn,
            patch(
                "backend.app.detect_drowsiness",
                return_value=("Alert", 0.9),
            ) as detect_drowsiness,
        ):
            response = self.request(
                "POST",
                "/detect",
                files={"file": ("driver.png", image_data.tobytes(), "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["yawn"]["status"], "No Yawn")
        for detector in (detect_yawn, detect_drowsiness):
            detector.assert_called_once()
            detector_input = detector.call_args.args[0]
            self.assertEqual(detector_input.shape, (96, 96, 3))
            np.testing.assert_array_equal(
                detector_input,
                np.zeros((96, 96, 3), dtype=np.uint8),
            )

    def test_health_remains_responsive_during_image_detection(self) -> None:
        encoded, image_data = cv2.imencode(
            ".png",
            np.zeros((96, 96, 3), dtype=np.uint8),
        )
        self.assertTrue(encoded)

        def slow_face_detection(frame):
            time.sleep(0.75)
            return frame, "No Face", False, [], None

        async def send_requests():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                started = asyncio.get_running_loop().time()
                detection_task = asyncio.create_task(
                    client.post(
                        "/detect",
                        files={
                            "file": (
                                "blank.png",
                                image_data.tobytes(),
                                "image/png",
                            )
                        },
                    )
                )
                await asyncio.sleep(0.05)
                health_response = await client.get("/health")
                health_elapsed = asyncio.get_running_loop().time() - started
                detection_response = await detection_task
                return health_response, health_elapsed, detection_response

        with patch(
            "backend.app.detect_driver_features",
            side_effect=slow_face_detection,
        ):
            health_response, health_elapsed, detection_response = asyncio.run(
                send_requests()
            )

        self.assertEqual(health_response.json(), {"status": "ok"})
        self.assertLess(health_elapsed, 0.5)
        self.assertTrue(detection_response.json()["success"])


if __name__ == "__main__":
    unittest.main()
