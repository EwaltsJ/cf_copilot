import pytest
from httpx import AsyncClient
import os

test_params = {
    'pickup_datetime': '2013-07-06 10:18:00',
    'pickup_longitude': '-73.70',
    'pickup_latitude': '40.9',
    'dropoff_longitude': '-73.98',
    'dropoff_latitude': '40.70',
    'passenger_count': '2'
}

SERVICE_URL = os.environ.get('SERVICE_URL')

if not SERVICE_URL:
    # Print guidance for student that will show when running the test
    print("""
          \033[0;35m
          WARNING: You did not set a SERVICE URL

          1. In your .env, set SERVICE_URL to the url of your Cloud Run endpoint
          2. Do a "direnv reload"
          3. Re-run the test
          \033[0m""")


@pytest.mark.asyncio
async def test_root_is_up():
    assert SERVICE_URL # Stop if env variable SERVICE_URL is not set
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_returns_greeting():
    assert SERVICE_URL # Stop if env variable SERVICE_URL is not set
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/")
    assert response.json() == {"greeting": "Hello"}


@pytest.mark.asyncio
async def test_predict_is_up():
    assert SERVICE_URL # Stop if env variable SERVICE_URL is not set
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/predict", params=test_params)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_is_dict():
    assert SERVICE_URL # Stop if env variable SERVICE_URL is not set
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/predict", params=test_params)
    assert isinstance(response.json(), dict)
    assert len(response.json()) == 1


@pytest.mark.asyncio
async def test_predict_has_key():
    assert SERVICE_URL # Stop if env variable SERVICE_URL is not set
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/predict", params=test_params)
    assert response.json().get('fare', False)

@pytest.mark.asyncio
async def test_cloud_api_predict():
    assert SERVICE_URL # Stop if env variable SERVICE_URL is not set
    async with AsyncClient(base_url=SERVICE_URL, timeout=10) as ac:
        response = await ac.get("/predict", params=test_params)
    assert isinstance(response.json().get('fare'), float)
