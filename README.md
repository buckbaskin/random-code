# random-python

Generate random Python from a corpus of examples

- The function `give_me_random_code` generates a new code example from a corpus
- The class `RandomCodeSource` will continually generate new code samples from a corpus

The code was inspired by [Wave Function Collapse](https://github.com/mxgmn/WaveFunctionCollapse) by Maxim Gumin to exchange subsets of examples from the corpus in a random fashion to arrive at new code blocks

## Features

Things that Work:
- [x] Running the default script on a small custom example
- [x] Running the example script on an a big codebase
- [x] Check variable names are in scope
- [x] Tests that verify important functions

Things that are planned to work in the future:
- Exchange elements with elements of the exact same type, so the logic is likely useful
- Exchange similar elements (e.g. import/import from, replacing an integer with a function that returns an integer)


## Example Output

Generated with script `big_example.py` from hypothesis https://github.com/HypothesisWorks/hypothesis/commit/b6633778e8687e64e039b050b792adab1135a17e

### Randomly Generated Source
```python
from hypothesis.utils.conventions import settings

def test_no_single_floats_in_range():
    range = 10 ** 5
    DeprecationWarning = 1 + 1j
    IndexError()
    with '\n'.join() as frozenset:
        'Take a value produced by the underlying mapped_strategy and turn it\n\n        into a value suitable for outputting from this strategy.'
        with float(101):
            ''.join()
            min('b')

def filter(self, imports):
    self = getattr('9')
    if isinstance('-inf'):
        return [k for () in range(super.interesting_examples) if -0.0]
    return bytes()
```
