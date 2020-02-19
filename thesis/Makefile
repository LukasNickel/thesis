all: build/thesis.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build
                                                                                
build/thesis.pdf: FORCE | build
	latexmk $(TeXOptions) thesis.tex


setup_plots:
	pdftk ../analysis/plots/cross_val_sep_perf_plot.pdf burst output cross_val_sep_perf_plot_%02d.pdf
	pdftk ../analysis/plots/cross_val_reg_perf_plot.pdf burst output cross_val_reg_perf_plot_%02d.pdf
	pdftk ../analysis/plots/cross_val_array_reg_perf_plot.pdf burst output cross_val_array_reg_perf_plot_%02d.pdf

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build
