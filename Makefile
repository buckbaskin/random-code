test:
	pytest test/

readme:
	python3 scripts/big_example.py >> README.md

.PHONY: test
