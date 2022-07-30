from random_code import give_me_random_code, find_files


def main():
    # Download and unzip the latest hypothesis branch into the folder corpus_hypothesis
    corpus_paths = find_files("corpus_hypothesis/hypothesis-master/hypothesis-python/")
    random_source = give_me_random_code(sorted(list(corpus_paths)))


if __name__ == "__main__":
    main()
