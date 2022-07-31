from random_code import find_files, give_me_random_code

def test_file_finding():
    corpus_paths = list(find_files("corpus/"))
    assert len(corpus_paths) > 0

def test_basics():
    corpus_paths = list(find_files("corpus/"))
    random_source = give_me_random_code(corpus_paths)
    try:
        eval(random_source)
    except SyntaxError:
        print('Random Source')
        print(random_source)
        print(repr(random_source))
        raise
