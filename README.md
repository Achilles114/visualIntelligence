# Visual Intelligence Platform

A Python visual-perception project for processing camera frames through object detection, multi-object tracking, scene processing, and monocular relative-depth estimation.

## Application overview

The runnable application is `apps/edge_runtime/detection_app.py`. It composes the platform’s processing components into a live camera pipeline:

1. Captures frames from camera source `0` with OpenCV.
2. Detects objects using TorchVision SSD Lite MobileNet V3 Large, with a confidence threshold of `0.5`.
3. Tracks detections between frames using an IoU tracker.
4. Estimates relative depth using MiDaS Small.
5. Builds scene results and displays them in an OpenCV window, including object counts, tracking labels, processing time, and recent events.
6. Prints frame and timing statistics when processing stops.

Press **q** in the preview window to stop the application. The current entry point uses a fixed camera source and graphical preview; it does not provide a headless mode, command-line camera selection, or HTTP server.

## Processing flow

```text
CameraSensor
    │ SensorFrame
    ▼
FramePipeline
    │
    ▼
ScenePipeline.process
    ├── PerceptionEngine
    │     └── TorchVision SSD Lite detector
    ├── TrackingEngine
    │     └── IoUTracker
    └── MiDaSSmallEstimator
    │
    ▼
SceneSnapshot
    │
    ▼
OpenCVPreview
```

The frame runtime manages camera opening and release, frame processing, optional result handling, and summary statistics. The edge application supplies the specific detector, tracker, scene processor, camera, and preview handler.

The default tracker configuration in the edge application is an IoU threshold of `0.3`, maximum age of `5`, and minimum hits of `2`.

## Technology stack

- **Python 3.11+** — project runtime requirement.
- **PyTorch and TorchVision** — model execution and object detection. The project configures the PyTorch CUDA 12.6 package index for uv.
- **OpenCV** — camera capture, image conversion, drawing, and GUI preview.
- **MiDaS Small** — monocular relative-depth estimation, loaded via `torch.hub`.
- **NumPy** — image and depth-map data.
- **FastAPI, Pydantic, and pydantic-settings** — declared dependencies; the current edge application does not start a web server.
- **uv** — dependency management using `pyproject.toml` and `uv.lock`.
- **pytest and Ruff** — development test and lint tools.

The detector and depth estimator select CUDA when PyTorch can access it; otherwise, they use CPU. GPU use in Docker additionally requires a compatible NVIDIA driver and NVIDIA Container Toolkit on the host.

## Repository layout

```text
apps/
  edge_runtime/
    detection_app.py       Edge application entry point and component wiring
    handlers/
      opencv_preview.py    Annotated OpenCV preview
configs/                   Configuration-related files
core/
  perception/              Detector interface, factory, and TorchVision detector
  reid/                    Re-identification components
  scene/                   Scene models and processing pipeline
  sensors/                 Camera sensor and frame models
  spatial/                 Geometry, motion, and depth processing
  tracking/                IoU and ReID tracking components
interfaces/                Project interface modules
runtime/
  frame_pipeline.py        Reusable frame-processing loop
tests/                     Unit and pipeline tests
main.py                    Minimal placeholder entry point
pyproject.toml             Project metadata and dependencies
uv.lock                    Resolved dependency lockfile
```

`main.py` is currently a placeholder that prints a greeting. Use `apps.edge_runtime.detection_app` to run the visual pipeline.

## Requirements

For a local run:

- Python 3.11 or newer
- A webcam or another OpenCV-compatible camera source
- Internet access for the first model downloads
- A graphical desktop session for the OpenCV preview

GPU acceleration is optional. CPU inference is selected automatically when CUDA is unavailable.

## Install and run locally

Install [uv](https://docs.astral.sh/uv/) if necessary, then run these commands from the repository root:

```bash
uv sync
uv run python -m apps.edge_runtime.detection_app
```

The detector loads TorchVision weights when it is created. The MiDaS estimator loads its model through `torch.hub` when the edge application is imported. First startup may take longer and requires internet access; later runs can reuse the local Torch cache.

Run the test suite:

```bash
uv run pytest
```

Run Ruff:

```bash
uv run ruff check .
```

## Docker

Docker packages the Python environment and application dependencies. Hardware access is platform-specific: the Compose example below is intended for a Linux host with `/dev/video0` and an X11 display. Docker Desktop on Windows or macOS does not generally expose a host webcam as `/dev/video0` to a Linux container. For those systems, run the application natively or provide a separate camera/display integration.

Build the image:

```bash
docker build -t visual-intelligence-platform .
```

### Linux webcam and preview with Compose

Allow local Docker containers to connect to the X server, then start the application:

```bash
xhost +local:docker
docker compose up --build
```

The Compose file maps `/dev/video0` into the container and mounts the X11 socket. If the camera is exposed at a different host device path, update the mapping in `docker-compose.yml`.

After stopping the application, revoke the display permission:

```bash
xhost -local:docker
```

### Linux webcam and preview with `docker run`

```bash
xhost +local:docker

docker run --rm -it \
  --device=/dev/video0:/dev/video0 \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v visual-intelligence-torch-cache:/root/.cache/torch \
  visual-intelligence-platform
```

The named volume persists downloaded model files between container runs.

### Optional NVIDIA GPU

GPU access requires a compatible host driver and the NVIDIA Container Toolkit. Add `--gpus all` to `docker run`:

```bash
docker run --rm -it --gpus all \
  --device=/dev/video0:/dev/video0 \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v visual-intelligence-torch-cache:/root/.cache/torch \
  visual-intelligence-platform
```

When CUDA is unavailable inside the container, PyTorch uses CPU.

### Docker limitations

- Linux webcam access requires mapping the host camera device into the container.
- The current preview handler requires a graphical display server.
- Docker Desktop on Windows and macOS does not provide the Linux webcam mapping shown above by default.
- Model downloads require network access on first use; the Torch cache volume avoids repeated downloads.
- The application currently has no headless mode or configurable camera-source option.

## Components and extension points

- `core/perception/base.py` defines the object-detector contract.
- `core/perception/factory.py` constructs detectors; the current factory supports `torchvision-ssd`.
- `core/perception/torchvision_detector.py` implements TorchVision SSD Lite MobileNet V3 Large inference.
- `core/tracking/` contains tracking interfaces, the IoU tracker, and ReID tracking components.
- `core/spatial/depth/` contains depth interfaces, models, and the MiDaS Small estimator.
- `core/scene/pipeline.py` composes perception, tracking, and depth processing for scene results.
- `core/sensors/camera.py` provides OpenCV camera capture.
- `runtime/frame_pipeline.py` provides the reusable frame-processing loop.
- `apps/edge_runtime/handlers/opencv_preview.py` renders scene results in an OpenCV window.
- `apps/edge_runtime/detection_app.py` wires the current edge application together.

## Troubleshooting

**The camera cannot be opened**

Confirm that the camera is connected and available to the host. In Linux Docker, verify that `/dev/video0` exists and is passed into the container.

**The preview window does not appear**

The current preview requires a graphical display. For Linux Docker, check `DISPLAY`, the X11 socket mount, and X server permissions. A headless container is not supported by the current preview handler.

**First startup is slow or model loading fails**

The TorchVision and MiDaS weights may need to download. Check internet access and ensure the Torch cache directory is writable. In Docker, the Compose configuration mounts a persistent cache volume.

**Inference is slow**

CPU inference is supported but may be slower. CUDA is selected only when PyTorch can access a compatible GPU. In Docker, verify the host driver, NVIDIA Container Toolkit, and GPU runtime configuration.