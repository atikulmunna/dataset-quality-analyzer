# syntax=docker/dockerfile:1.7

# Pin the linux/amd64 manifest used by the initial us-east-1 Fargate worker.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941

FROM ${PYTHON_IMAGE} AS dependencies
COPY requirements/worker.lock /tmp/worker.lock
RUN python -m pip install \
      --disable-pip-version-check \
      --no-cache-dir \
      --no-compile \
      --target /opt/python \
      --requirement /tmp/worker.lock

FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="Dataset Quality Analyzer worker" \
      org.opencontainers.image.source="https://github.com/atikulmunna/dataset-quality-analyzer" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/dqa:/opt/python \
    HOME=/tmp

RUN groupadd --gid 10001 dqa \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin dqa \
    && python -m pip uninstall --yes pip setuptools wheel \
    && rm -rf /root/.cache /usr/local/lib/python3.11/site-packages/pip* /usr/local/lib/python3.11/site-packages/setuptools*

COPY --from=dependencies /opt/python /opt/python
COPY --chown=10001:10001 dqa /opt/dqa/dqa
COPY --chown=10001:10001 dqa.yaml dqa_seg.yaml dqa_seg_low_noise.yaml /opt/dqa/configs/

WORKDIR /workspace
USER 10001:10001

# AWS Batch treats this as a finite job: exit status is its health signal.
ENTRYPOINT ["python", "-m", "dqa.worker_entry"]
CMD ["--help"]
