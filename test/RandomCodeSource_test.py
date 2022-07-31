from random_code import find_files, RandomCodeSource


def exec_helper(random_source):
    try:
        exec(random_source)
    except SyntaxError:
        print("Random Source")
        print(random_source)
        print(repr(random_source))
        raise
    except TypeError:
        print("Random Source")
        print(random_source)
        print(repr(random_source))
        raise


def test_RandomCodeSource_seed(integer):
    corpus_paths = list(find_files("corpus/"))
    code_generator = RandomCodeSource(
        corpus_paths, seed=1234 + integer, prettyprinter=False
    )

    random_source = code_generator.next_source()

    exec_helper(random_source)


def test_RandomCodeSource_sequence(integer):
    corpus_paths = list(find_files("corpus/"))
    code_generator = RandomCodeSource(corpus_paths, seed=1234, prettyprinter=True)

    for i in range(0, integer):
        random_source = code_generator.next_source()

    exec_helper(random_source)


def pytest_generate_tests(metafunc):
    if "integer" in metafunc.fixturenames:
        metafunc.parametrize("integer", range(1, 5))
