FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY predict.py model.bin ./

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 9696

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]