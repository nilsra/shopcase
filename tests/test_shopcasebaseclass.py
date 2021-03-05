
from pathlib import Path

from shopcasebaseclass import ShopCaseBaseClass

DEFAULT_DIFF_OF_COPY = "{}"



def _load_testcase():
    """ Return basic.py as a ShopCaseBaseClass instance. """
    return ShopCaseBaseClass(Path(__file__).parent / 'data')


def test_case_from_yaml_files():
    s = _load_testcase()
    assert isinstance(s, ShopCaseBaseClass)


def test_copy():
    s1 = _load_testcase()
    s2 = s1.copy()
    assert str(s1.diff(s2)) == DEFAULT_DIFF_OF_COPY


def test_copy_and_diff():
    s1 = _load_testcase()
    s2 = s1.copy()
    s2.case['model']['reservoir']['Reservoir1']['inflow'] += 1
    assert str(s1.diff(s2)) == "{'model': {'reservoir': {'Reservoir1': {'inflow': True}}}}"


def test_run():
    s = _load_testcase()
    s.run()
    status = s.case['model']['objective']['average_objective']['solver_status']
    assert status == 'Optimal solution is available'


def test_to_json():
    s1 = _load_testcase()
    s2 = ShopCaseBaseClass(s1.to_json())
    assert str(s1.diff(s2)) == DEFAULT_DIFF_OF_COPY


def test_to_bytestring():
    s1 = _load_testcase()
    s2 = ShopCaseBaseClass(s1._to_bytestring())
    assert str(s1.diff(s2)) == DEFAULT_DIFF_OF_COPY
    