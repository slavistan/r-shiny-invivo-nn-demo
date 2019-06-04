.PHONY: run

run:
	R -q -e 'rmarkdown::run("index.rmd", shiny_args=list(port=1337))'
