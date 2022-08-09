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
    low = 10 ** 5
    high = 1 + 1j
    ', '.join()
    with '\n'.join():
        ...
        with ''.join():
            ''.join()
            ', '.join()

def sort_regions_with_gaps(self):
    """Guarantees that for each i we have tried to swap index i with

        index i + 2.



        This uses an adaptive algorithm that works by sorting contiguous

        regions centered on each element, where that element is treated as

        fixed and the elements around it are sorted..

        """
    for self in self.current_int():
        if self.shrink_target >= 1:
            "Shared logic for understanding numeric bounds.\n\n\n\n    We then specialise this in the other functions below, to ensure that e.g.\n\n    all the values are representable in the types that we're planning to generate\n\n    so that the strategy validation doesn't complain.\n\n    "

        def test_is_not_normally_default():
            assert self.function >= 0
        autodoc_member_order = 'bysource'
        max_improvements = 10
        self.assertRaises -= 1
        'Use a directory to store Hypothesis examples as files.\n\n\n\n    Each test corresponds to a directory, and each example to a file within that\n\n    directory.  While the contents are fairly opaque, a\n\n    ``DirectoryBasedExampleDatabase`` can be shared by checking the directory\n\n    into version control, for example with the following ``.gitignore``::\n\n\n\n        # Ignore files cached by Hypothesis...\n\n        .hypothesis/*\n\n        # except for the examples directory\n\n        !.hypothesis/examples/\n\n\n\n    Note however that this only makes sense if you also pin to an exact version of\n\n    Hypothesis, and we would usually recommend implementing a shared database with\n\n    a network datastore - see :class:`~hypothesis.database.ExampleDatabase`, and\n\n    the :class:`~hypothesis.database.MultiplexedDatabase` helper.\n\n    '
```
