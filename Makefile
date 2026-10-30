.PHONY: verify figures selftest regulatory-selftest protein-selftest clean-generated

verify:
	python verify_results.py

figures:
	python figures/make_figures.py

regulatory-selftest:
	python regulatory/g1b_reb1_independent_null.py --self-test
	python regulatory/zhu2014_mutation_model.py --self-test
	python regulatory/fit_large_library_models.py --self-test
	python regulatory/build_g1e_positions.py --self-test
	python regulatory/mutability_gate.py --self-test

protein-selftest:
	python protein/family_quality_screen.py --self-test
	python protein/meff_bound.py --self-test

selftest: regulatory-selftest protein-selftest

clean-generated:
	rm -rf figures/generated
