# Portable container for any host that takes a Dockerfile: Google Cloud Run,
# Koyeb, Fly.io, a VPS, or a Hugging Face Docker Space on a paid plan.
#
# Defaults to the Streamlit build on $PORT, which is what Cloud Run expects.
# For the Gradio build, override the command:
#   docker run -e GRADIO_SERVER_NAME=0.0.0.0 -e GRADIO_SERVER_PORT=8080 \
#              -p 8080:8080 carecompass python app.py

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    # No torch in the image, so skip the import attempt at startup.
    CARECOMPASS_DISABLE_DENSE=1

WORKDIR /app

# Dependencies first so edits to the application do not invalidate the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run unprivileged, and give the process a writable home for the event log.
RUN useradd --create-home --uid 1000 carecompass \
    && mkdir -p /app/logs /app/data/index \
    && chown -R carecompass:carecompass /app
USER carecompass

EXPOSE 8080

# Fail the container health check if the corpus or the rules cannot load.
HEALTHCHECK --interval=60s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from src.pipeline import CareCompass; CareCompass()" || exit 1

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port ${PORT} --server.address 0.0.0.0"]
