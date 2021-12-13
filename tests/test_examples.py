import os
import subprocess
import sys

import pytest

EXAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")
)
EXAMPLES = [
    "ising_model.py",
    #  "rbm.py",
]


@pytest.mark.parametrize("example", EXAMPLES)
def test_example(example):
    print(f"Running:\npython examples/{example}")
    filename = os.path.join(EXAMPLES_DIR, example)
    subprocess.check_call([sys.executable, filename])
