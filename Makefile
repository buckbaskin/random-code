test:
	pytest test/

readme:
	cat README.mdt > README.md
	python3 scripts/big_example.py >> README.md

.PHONY: test
