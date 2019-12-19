all: build/presentation.pdf

texoptions = \
	     --lualatex \
	     --interaction=nonstopmode \
	     --halt-on-error \
	     --output-directory=build

build/presentation.pdf: FORCE | build
	latexmk $(texoptions) presentation.tex

preview: FORCE | build
	latexmk $(texoptions) -pvc presentation.tex

FORCE:

build:
	mkdir -p build

clean:
	rm -r build
