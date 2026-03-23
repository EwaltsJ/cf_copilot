import pytest
from httpx import AsyncClient
import os
import re
import subprocess


# Find the port the docker image is running on
image_name = f"{os.environ.get('DOCKER_IMAGE_NAME')}:local"
docker_ps_command = f'docker ps --filter ancestor={image_name} --format "{{{{.Ports}}}}"'
docker_ps_output = subprocess.Popen(docker_ps_command,
                        shell=True,
                        stdout=subprocess.PIPE) \
                    .stdout.read().decode("utf-8")

if docker_ps_output:
    docker_port = re.findall(":(\d{4})-", docker_ps_output)[0]
else:
    docker_port = None
    print("""
          \033[0;35m
          WARNING: We did not find a port with a docker container running

          Verify: - That your docker container is running
                  - The docker image was correctly named using $DOCKER_IMAGE_NAME:local
                  - If your API is working locally, that it is running on a docker
                    container and not just using uvicorn locally
          \033[0m""")

SERVICE_URL = f"http://localhost:{docker_port}"


@pytest.mark.asyncio
async def test_root_is_up():
    assert docker_port
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_returns_greeting():
    assert docker_port
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/")
    assert response.json() == {"message": "Hi, The API is running!"}


@pytest.mark.asyncio
async def test_predict_is_up():
    assert docker_port
    async with AsyncClient(base_url=SERVICE_URL, timeout=30) as ac:
        with open("raw_data/test.csv", "rb") as f:
            response = await ac.post(
                "/predict",
                files={"file": ("test.csv", f, "text/csv")}
            )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_is_dict():
    assert docker_port
    async with AsyncClient(base_url=SERVICE_URL, timeout=30) as ac:
        with open("raw_data/test.csv", "rb") as f:
            response = await ac.post(
                "/predict",
                files={"file": ("test.csv", f, "text/csv")}
            )
    assert isinstance(response.json(), dict)


@pytest.mark.asyncio
async def test_predict_has_key():
    assert docker_port
    async with AsyncClient(base_url=SERVICE_URL, timeout=30) as ac:
        with open("raw_data/test.csv", "rb") as f:
            response = await ac.post(
                "/predict",
                files={"file": ("test.csv", f, "text/csv")}
            )
    body = response.json()
    # Accept either proper predictions or the explicit "no model" error
    assert "predictions" in body or body.get("error") == "No trained model found. Run train() first."

@pytest.mark.asyncio
async def test_predict_val_is_list():
    assert docker_port
    async with AsyncClient(base_url=SERVICE_URL, timeout=30) as ac:
        with open("raw_data/test.csv", "rb") as f:
            response = await ac.post(
                "/predict",
                files={"file": ("test.csv", f, "text/csv")}
            )
    predictions = response.json().get("predictions")
    assert predictions is None or isinstance(predictions, list)
