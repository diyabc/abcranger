import pytest
import os


def pytest_addoption(parser):
    parser.addoption("--path", action="store", default="executable path")

@pytest.fixture
def path(request):
    return request.config.getoption("--path")

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)),"data"))
