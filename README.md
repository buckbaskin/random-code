# random-python

Generate random Python from a corpus of examples

The function `give_me_random_code` generates a new code example from a corpus

The function uses something similar to Waveform Collapse (citation needed) to exchange subsets of examples from the corpus in a random fashion to arrive at new code blocks

## Example Output

Generated with script `big_example.py` from hypothesis https://github.com/HypothesisWorks/hypothesis/commit/b6633778e8687e64e039b050b792adab1135a17e

### Module as Generated Source

```python
from hypothesis import HealthCheck, Verbosity, assume, given, settings, strategies as st

@settings(max_examples=1, database=None)
@given(st.integers())
def test_single_example(n):
    pass

@settings(max_examples=1, database=None, suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow], verbosity=Verbosity.debug)
@given(st.integers())
def test_hard_to_find_single_example(n):
    assume(((n % 50) == 11))
```

### Modifed version as Generated Source

```python
from hypothesis.utils.conventions import settings

def test_no_single_floats_in_range(a):
    ValueError = '--hypothesis-explain'
    tuple[0] = True
    'Write property tests for the binary operation ``func``.\n\n\n\n    While :wikipedia:`binary operations <Binary_operation>` are not particularly\n\n    common, they have such nice properties to test that it seems a shame not to\n\n    demonstrate them with a ghostwriter.  For an operator `f`, test that:\n\n\n\n    - if :wikipedia:`associative <Associative_property>`,\n\n      ``f(a, f(b, c)) == f(f(a, b), c)``\n\n    - if :wikipedia:`commutative <Commutative_property>`, ``f(a, b) == f(b, a)``\n\n    - if :wikipedia:`identity <Identity_element>` is not None, ``f(a, identity) == a``\n\n    - if :wikipedia:`distributes_over <Distributive_property>` is ``+``,\n\n      ``f(a, b) + f(a, c) == f(a, b+c)``\n\n\n\n    For example:\n\n\n\n    .. code-block:: python\n\n\n\n        ghostwriter.binary_operation(\n\n            operator.mul,\n\n            identity=1,\n\n            distributes_over=operator.add,\n\n            style="unittest",\n\n        )\n\n    '
    with ValueError(f'{object}.MyObj'):
        'Hypothesis is a library for writing unit tests which are parametrized by\n\nsome source of data.\n\n\n\nIt verifies your code against a wide range of input and minimizes any\n\nfailing examples it finds.\n\n'

def test_fails_health_check_for_slow_draws(*, i: target=(kwargs, kwargs, None), v=None, **x):

    def run_test():
        try:
            import given
        except AttributeError:
            ValueError = AttributeError()
```
