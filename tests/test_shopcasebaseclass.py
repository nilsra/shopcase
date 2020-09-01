
from pathlib import Path

from shopcasebaseclass import ShopCaseBaseClass


def _load_testdata():
    """ Returning basic.py as a JSON string. """
    with open(Path(__file__).parent / 'testmodel_basic.json', 'r') as f:
        return f.read()


def _load_testcase():
    """ Return basic.py as a ShopCaseBaseClass instance. """
    return ShopCaseBaseClass(_load_testdata())


def test_case_from_json():
    s = ShopCaseBaseClass(_load_testdata)
    assert isinstance(s, ShopCaseBaseClass)


def test_copy():
    s1 = _load_testcase()
    s2 = s1.copy()
    assert (s1 is not s2) and (s1.case is not s2.case)


def test_diff():
    s1 = _load_testcase()
    s2 = s1.copy()
    s2.case['model']['reservoir']['Reservoir1']['inflow'] += 1
    assert str(s1.diff(s2)) == "{'model': {'reservoir': {'Reservoir1': {'inflow': True}}}}"


def test_copy2():
    s1 = _load_testcase()
    s2 = s1.copy()
    assert s1.diff(s2, tolerance=0.0001) == {}


def test_run():
    s = _load_testcase()
    s.run()
    status = s.case['model']['objective']['average_objective']['solver_status']
    assert status == 'Optimal solution is available'


def test_to_bytestring():
    s1 = _load_testcase()
    s2 = ShopCaseBaseClass(s1._to_bytestring())
    assert s1.diff(s2, tolerance=0.0001) == {}
    