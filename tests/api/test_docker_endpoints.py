import pytest
from httpx import AsyncClient
import os
import re
import subprocess

test_params = {
    "input_one": "5.0",
    "input_two": "10.0",
}

# Find the port the docker image is running on
image_name = f"{os.environ.get('GAR_IMAGE')}:dev"
# Use docker ps to list all running containers derived from $GAR_IMAGE:dev
docker_ps_command = f'docker ps --filter ancestor={image_name} --format "{{{{.Ports}}}}"'
docker_ps_output = subprocess.Popen(docker_ps_command,
                        shell=True,
                        stdout=subprocess.PIPE) \
                    .stdout.read().decode("utf-8")

# If we have an output, extract the port the container is running on
if docker_ps_output:
    docker_port = re.findall(":(\d{4})-", docker_ps_output)[0]
else:
    # If no output set docker_port to None
    # In the tests we'll assert docker_port exists
    docker_port = None
    # Print guidance for student that will show when running the test
    print("""
          \033[0;35m
          WARNING: We did not find a port with a docker container running

          Verify: - That your docker container is running
                  - The docker image was correctly named using $GAR_IMAGE:dev
                  - If your API is working locally, that it is running on a docker
                    container and not just using uvicorn locally
          \033[0m""")

# Assemble the service url
SERVICE_URL = f"http://localhost:{docker_port}"

@pytest.mark.asyncio
async def test_root_is_up():
    assert docker_port # Stop if no docker port found
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_returns_greeting():
    assert docker_port # Stop if no docker port found
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/")
    assert response.json() == {'message': "Hi, The API is running!"}


@pytest.mark.asyncio
async def test_predict_is_up():
    assert docker_port # Stop if no docker port found
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/predict", params=test_params)
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_predict_is_dict():
    assert docker_port
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/predict", params=test_params)
    body = response.json()
    assert isinstance(body, dict)
    assert set(body.keys()) == {"prediction", "inputs"}

@pytest.mark.asyncio
async def test_predict_is_dict2():
    assert docker_port # Stop if no docker port found
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/predict", params=test_params)
    assert isinstance(response.json(), dict)
    assert len(response.json()) == 2

# run tbe below tests when the api is actually returning something from mlflow setup
"""
@pytest.mark.asyncio
async def test_predict_has_key():
    assert docker_port # Stop if no docker port found
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/predict", params=test_params)
    assert response.json().get('fare', False)

@pytest.mark.asyncio
async def test_cloud_api_predict():
    assert docker_port # Stop if no docker port found
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/predict", params=test_params)
    assert isinstance(response.json().get('fare'), float)
"""
