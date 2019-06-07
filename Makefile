.PHONY: run

run:
	R -q -e 'rmarkdown::run("index.rmd", shiny_args=list(port=1337))'

render:
	R -q -e 'rmarkdown::render("index.rmd", output_dir="output")'
