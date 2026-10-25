# Shared image for the API and the Streamlit app; EXTRA selects the dependency set.
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

ARG EXTRA
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /srv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra ${EXTRA}
COPY src ./src
RUN uv sync --frozen --no-dev --extra ${EXTRA}

RUN useradd --create-home appuser
USER appuser
