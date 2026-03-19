FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY cf_copilot/ cf_copilot/
COPY setup.py .
RUN pip install --no-cache-dir -e .

# Port exposed by Cloud Run / your hosting provider
ENV PORT=8080
ENV LOCAL_REGISTRY_PATH=/app/cf_copilot
EXPOSE 8080

CMD ["uvicorn", "cf_copilot.api.fast:app", "--host", "0.0.0.0", "--port", "8080"]
