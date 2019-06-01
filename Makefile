.PHONY: run

run:
	R -q -e 'rmarkdown::run("index.rmd")'
