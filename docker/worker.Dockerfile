# syntax=docker/dockerfile:1.7

# Pin the linux/amd64 manifest. Alpine avoids the unused Perl runtime that
# carried unfixed critical/high findings in the previous Debian base.
ARG PYTHON_IMAGE=python:3.11.15-alpine3.24@sha256:bbc78bdb39abbac9225c0f50643c4313e6b06ba1cf7d1dcc34249f13cecaa3d7

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

RUN addgroup -g 10001 dqa \
    && adduser -D -H -u 10001 -G dqa -s /sbin/nologin dqa \
    && install -d -o 10001 -g 10001 -m 0700 /workspace \
    && python -m pip uninstall --yes pip setuptools wheel \
    && rm -rf /root/.cache /usr/local/lib/python3.11/site-packages/pip* /usr/local/lib/python3.11/site-packages/setuptools*

COPY --from=dependencies /opt/python /opt/python
COPY --chown=10001:10001 dqa /opt/dqa/dqa
COPY --chown=10001:10001 dqa.yaml dqa_seg.yaml dqa_seg_low_noise.yaml /opt/dqa/configs/
COPY --chown=0:0 docker/worker_bootstrap.py /opt/dqa/worker_bootstrap.py

WORKDIR /workspace

# The bootstrap only fixes the Fargate volume owner, then irreversibly drops to
# UID/GID 10001 before importing or executing application code.
ENTRYPOINT ["python", "/opt/dqa/worker_bootstrap.py"]
CMD ["--help"]
