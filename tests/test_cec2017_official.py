"""The ctypes wrapper of the official CEC2017 code against a standalone build of the same code."""
import shutil
import subprocess

import numpy as np
import pytest

from bioqpso.benchmarks import cec2017_official as co

POINTS = 5
SEED = 20261106


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    """Standalone program: the official function file plus a main(), run next to input_data/."""
    d = tmp_path_factory.mktemp("cec17")
    exe = d / "driver"
    subprocess.run(["c++", "-O2", "-w", "-o", str(exe), str(co.HERE / "driver.cpp"),
                    str(co.HERE / "cec17_test_func.cpp")], check=True)
    (d / "input_data").symlink_to(co.DATA)
    return exe


def standalone(exe, k, x):
    r = subprocess.run([str(exe), str(k), str(x.shape[1]), str(x.shape[0])], cwd=exe.parent, check=True,
                       input=" ".join(f"{v:.17g}" for v in x.ravel()), capture_output=True, text=True)
    return np.array([float(v) for v in r.stdout.split()])


def test_suite():
    assert len(co.FUNCS) == 29 and 2 not in co.FUNCS


@pytest.mark.parametrize("D", co.DIMS)
@pytest.mark.parametrize("k", co.FUNCS)
def test_wrapper_equals_standalone(driver, k, D):
    x = np.random.default_rng(SEED + D).uniform(-100, 100, (POINTS, D))
    ours, ref = co.evaluate(k, x), standalone(driver, k, x)
    assert np.all(np.abs(ours - ref) <= 1e-12 * np.abs(ref))
    # one point at a time, as the optimizers call it
    assert np.all(np.abs(np.array([co.OfficialFunction(k, D)(xi) for xi in x]) - ref) <= 1e-12 * np.abs(ref))


@pytest.mark.parametrize("D", co.DIMS)
@pytest.mark.parametrize("k", co.FUNCS)
def test_bias_at_optimum(k, D):
    assert co.evaluate(k, co.optimum(k, D))[0] == pytest.approx(100.0 * k, abs=1e-8)


def test_problem_dict():
    p = co.get_official_problem("F7", 30)
    assert (p["lb"], p["ub"], p["dim"], p["f_opt"], p["name"]) == (-100.0, 100.0, 30, 700.0, "F7")
    assert p["func"](co.optimum(7, 30)) == pytest.approx(700.0)


def test_rejects_f2():
    with pytest.raises(ValueError):
        co.evaluate(2, np.zeros(30))


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
def test_pickles():
    import pickle
    f = pickle.loads(pickle.dumps(co.OfficialFunction(3, 50)))
    assert f(co.optimum(3, 50)) == pytest.approx(300.0)
