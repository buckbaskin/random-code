from random_code import find_files, give_me_random_code, RandomCodeSource


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


def test_file_finding():
    corpus_paths = list(find_files("corpus/"))
    assert len(corpus_paths) > 0


def test_basics():
    corpus_paths = list(find_files("corpus/"))
    random_source = give_me_random_code(corpus_paths).strip()
    exec_helper(random_source)


def test_RandomCodeSource(integer):
    corpus_paths = list(find_files("corpus/"))
    code_generator = RandomCodeSource(corpus_paths, seed=1234, prettyprinter=False)

    for i in range(0, integer):
        random_source = code_generator.next_source()

    exec_helper(random_source)


def pytest_generate_tests(metafunc):
    if "integer" in metafunc.fixturenames:
        metafunc.parametrize("integer", range(1, 5))
