.PHONY: run
run:
	R -q -e 'rmarkdown::run("index.rmd", shiny_args=list(port=1337))'

.PHONY: render
render:
	R -q -e 'rmarkdown::render("index.rmd", output_dir="output")'
	firefox.exe "output/index.html"
