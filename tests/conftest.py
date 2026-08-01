import torch
import pytest


@pytest.fixture
def identity_c2w():
    return torch.eye(4, dtype=torch.float32)
