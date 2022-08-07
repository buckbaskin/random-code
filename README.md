# random-python

Generate random Python from a corpus of examples

- The function `give_me_random_code` generates a new code example from a corpus
- The class `RandomCodeSource` will continually generate new code samples from a corpus

The function uses something similar to Waveform Collapse (citation needed) to exchange subsets of examples from the corpus in a random fashion to arrive at new code blocks

## Features

Things that Work:
- [x] Running the default script on a small custom example

Things that maybe work:
- [ ] Running the example script on an a big codebase
- [ ] Check variable names are in scope
- [ ] Tests that verify important functions

Things that are planned to work in the future:
- Exchange elements with elements of the exact same type, so the logic is likely useful
- Exchange similar elements (e.g. import/import from, replacing an integer with a function that returns an integer)


## Example Output

Generated with script `big_example.py` from hypothesis https://github.com/HypothesisWorks/hypothesis/commit/b6633778e8687e64e039b050b792adab1135a17e

### Randomly Generated Source
```python

from hypothesis.utils.conventions import settings

def test_no_single_floats_in_range():
    low = (10 ** 5)
    high = (1 + 1j)
    ', '.join()
    with pytest.raises(InvalidArgument):
        "A wrapper to make the given database read-only.\n\n\n\n    The implementation passes through ``fetch``, and turns ``save``, ``delete``, and\n\n    ``move`` into silent no-ops.\n\n\n\n    Note that this disables Hypothesis' automatic discarding of stale examples.\n\n    It is designed to allow local machines to access a shared database (e.g. from CI\n\n    servers), without propagating changes back from a local or in-development branch.\n\n    "
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            st.floats(low, high, width=32).validate()

def test_subTest():
    suite = '\n\ntry:\n\n{}\n\n    exc_type = None\n\n    target(1, label="input was valid")\n\n{}except Exception as exc:\n\n    exc_type = type(exc)\n\n'.strip()
    ', '.join((0.0, 5e-324))
    stream = '\n'.join()
    out = ', '.join()
    assert ((n_value,) in (2, 1)), out

```
