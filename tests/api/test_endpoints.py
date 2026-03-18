import pytest
from httpx import AsyncClient


test_params = {
    'input_one': '5.0',
    'input_two': '10.0'
}


@pytest.mark.asyncio
async def test_root_is_up():
    from packagename.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_returns_greeting():
    from packagename.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.json() == {"message": "Hi, The API is running!"}


@pytest.mark.asyncio
async def test_predict_is_up():
    from packagename.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict", params=test_params)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_is_dict():
    from packagename.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict", params=test_params)
    body = response.json()
    assert isinstance(body, dict)
    assert set(body.keys()) == {"prediction", "inputs"}


@pytest.mark.asyncio
async def test_predict_has_key():
    from packagename.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict", params=test_params)
    assert "prediction" in response.json()


@pytest.mark.asyncio
async def test_predict_val_is_float():
    from packagename.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict", params=test_params)
    assert isinstance(response.json().get("prediction"), float)
    assert response.json().get("prediction") == 15.0
