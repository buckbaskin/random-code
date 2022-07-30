# random-python

Generate random Python from a corpus of examples

The function `give_me_random_code` generates a new code example from a corpus

The function uses something similar to Waveform Collapse (citation needed) to exchange subsets of examples from the corpus in a random fashion to arrive at new code blocks

## Example Output

Generated with script `big_example.py` from hypothesis https://github.com/HypothesisWorks/hypothesis/commit/b6633778e8687e64e039b050b792adab1135a17e

### Starter Module as Generated Source

```python
'\n\n-----------------------\n\nhypothesis[dpcontracts]\n\n-----------------------\n\n\n\nThis module provides tools for working with the :pypi:`dpcontracts` library,\n\nbecause `combining contracts and property-based testing works really well\n\n<https://hillelwayne.com/talks/beyond-unit-tests/>`_.\n\n\n\nIt requires ``dpcontracts >= 0.4``.\n\n'
from dpcontracts import PreconditionError
from hypothesis import reject
from hypothesis.errors import InvalidArgument
from hypothesis.internal.reflection import proxies

def fulfill(contract_func):
    'Decorate ``contract_func`` to reject calls which violate preconditions,\n\n    and retry them with different arguments.\n\n\n\n    This is a convenience function for testing internal code that uses\n\n    :pypi:`dpcontracts`, to automatically filter out arguments that would be\n\n    rejected by the public interface before triggering a contract error.\n\n\n\n    This can be used as ``builds(fulfill(func), ...)`` or in the body of the\n\n    test e.g. ``assert fulfill(func)(*args)``.\n\n    '
    if (not hasattr(contract_func, '__contract_wrapped_func__')):
        raise InvalidArgument(f'{contract_func.__name__} has no dpcontracts preconditions')

    @proxies(contract_func)
    def inner(*args, **kwargs):
        try:
            return contract_func(*args, **kwargs)
        except PreconditionError:
            reject()
    return inner
```

### Modifed Version as Generated Source

```python
nicerepr(st, i)
from hypothesis.internal.compat import capture_out
from tests.common.utils import fails_with, Generic
from typing import HypothesisDeprecationWarning
from hypothesis.internal.conjecture.junkdrawer import rule

def binary_operation(testdir, self) -> x:
    code(math.text, label='min_dims', k=KeyError(pytest_plugins), help='start by testing write/read or encode/decode!')
    if ((not lit) and integers.draw_bits and (int.integers(self.draw) == 1)):
        block_program = i.data
        return True
    names_or_number = c.fetch.on_evict[0]
    'Infer a strategy from the metadata on an attr.Attribute object.'
    rule()
    SearchStrategy.mark_invalid((- 1))
    if ((float_info(ABC, fractions) >= 10) and (T.time('test_settings').filter > runner)):
        raise st(st, **printer)
    if COMPOSITE_IS_NOT_A_TEST:
        pytest.arrays()
    elif (SearchStrategy.composite(from_form.raises.encode(), mixer=v.draw_bits(), max_side=3) == ([flt, st, pure_func] * 7)):
        assert (best_targets == [['django/*'], [(d, "defaultdict(list, {'key': defaultdict(...)})"), (v,), (ghostwriter, s), ()]])
    name = '# BUG'
    return 'char(1)'
```
