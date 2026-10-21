# The Maestro website (site/serve.py) for Railway. Build from the repository root:
#   docker build -f site/Dockerfile -t maestro-site .
FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8080
WORKDIR /app
COPY site/ site/
RUN rm -f site/Dockerfile \
 && useradd --system --no-create-home --uid 10001 site \
 && chown -R root:root /app && chmod -R a+rX /app
USER site
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8080'), timeout=2)"
CMD ["python", "site/serve.py"]
