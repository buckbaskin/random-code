from random_code import find_files, give_me_random_code, RandomCodeSource


def test_file_finding():
    corpus_paths = list(find_files("corpus/"))
    assert len(corpus_paths) > 0


def test_basics():
    corpus_paths = list(find_files("corpus/"))
    random_source = give_me_random_code(corpus_paths)
    try:
        eval(random_source)
    except SyntaxError:
        print("Random Source")
        print(random_source)
        print(repr(random_source))
        raise


def test_RandomCodeSource(integer):
    corpus_paths = list(find_files("corpus/"))
    code_generator = RandomCodeSource(corpus_paths, seed=1234)

    for i in range(0, integer):
        random_source = code_generator.next_source()

        try:
            eval(random_source)
        except SyntaxError:
            print("Random Source")
            print(random_source)
            print(repr(random_source))
            raise


def pytest_generate_tests(metafunc):
    if "integer" in metafunc.fixturenames:
        metafunc.parametrize("integer", range(5))
