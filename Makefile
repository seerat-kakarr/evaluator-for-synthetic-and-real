PYTHON ?= .venv/bin/python
DESCRIPTION ?= Ardook is an Armenia-based social enterprise providing flexible work-from-home laundry jobs to single mothers caring for children with disabilities.
ATTRIBUTES ?= {"region":"Armenia","entity_type":"social enterprise"}

.PHONY: help tracker synthetic recommend recommend-synthetic

help:
	@echo "make tracker             Evaluate rankings against the real grant tracker"
	@echo "make synthetic           Evaluate rankings on generated data"
	@echo "make recommend           Show real-tracker recommendations for Ardook"
	@echo "make recommend-synthetic Show recommendations for a sample generated profile"

tracker:
	$(PYTHON) -m src.evaluate_relevance --dataset tracker

synthetic:
	$(PYTHON) -m src.evaluate_relevance --dataset synthetic

recommend:
	$(PYTHON) -m src.recommend --dataset real --description "$(DESCRIPTION)" --attributes '$(ATTRIBUTES)'

recommend-synthetic:
	$(PYTHON) -m src.recommend --dataset synthetic --profile-id 1
