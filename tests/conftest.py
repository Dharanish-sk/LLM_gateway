import pytest

from gateway.config import load_settings


@pytest.fixture
def settings():
    config = load_settings("config/demo.yaml")
    config.bucket_capacity = 10000
    config.bucket_refill_rate = 10000
    return config
