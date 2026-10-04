FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml constraints.txt ./
COPY gateway ./gateway
COPY experiments ./experiments
RUN pip install --no-cache-dir -c constraints.txt . && useradd --uid 10001 --create-home gateway
COPY config ./config
COPY data ./data
USER gateway
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready',timeout=2)"
CMD ["gateway", "serve", "--host", "0.0.0.0"]
