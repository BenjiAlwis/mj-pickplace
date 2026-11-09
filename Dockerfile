# Reproducible, headless environment for tests, benchmarks and rendering.
FROM python:3.12-slim

# git: fetch Menagerie; libegl/libgl: headless MuJoCo rendering (MUJOCO_GL=egl).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates libegl1 libgl1 libglvnd0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY scripts/fetch_assets.sh scripts/
RUN bash scripts/fetch_assets.sh

COPY pyproject.toml README.md ./
COPY pnp pnp
RUN pip install --no-cache-dir -e ".[dev,media]"

COPY tests tests
COPY scripts scripts

# Default: run the test suite. Override, e.g.
#   docker run --rm -v $PWD/results:/app/results pnp python -m pnp.eval.run_eval --n 200
CMD ["pytest", "-q"]
