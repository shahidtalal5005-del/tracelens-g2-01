import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tracelens.agent import Agent
from tracelens.eval.bench import build_bench
from tracelens.world import World


@pytest.fixture(scope="session")
def world():
    return World(7)


@pytest.fixture(scope="session")
def agent(world):
    return Agent(world)


@pytest.fixture(scope="session")
def tasks(world):
    return world.make_tasks(3)


@pytest.fixture(scope="session")
def bench():
    return build_bench(7, tasks_per_type=4)
