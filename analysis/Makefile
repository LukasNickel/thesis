prep_f = preprocessed_data
merged_f = merged_data
models_f = ml_models
dl2_f = dl2
plots_f = plots
scripts_f = software/scripts

merge_script = software/cta_preprocessing/merge_files.py
prep_script = software/cta_preprocessing/process_multiple_simtel_files.py

tel_conf = config/aict_tel_config.yaml
array_conf = config/aict_array_config.yaml
tel_conf_mono = config/aict_tel_config_mono.yaml
selection_conf = config/aict_selection.yaml

.PHONY: merge_data split_data analysis plots all clean

from_scratch: directories install preprocessing analysis thesis

all: preprocessing analysis plots thesis

test:
	TEXINPUTS=$$(pwd): \
	python test_pgf.py

analysis: directories merge_data split_data train_models dl2 plots cut cut_plots

# not removing prep folder 
clean:
	rm -r $(merged_f)
	rm -r $(models_f)
	rm -r $(dl2_f)
	rm -r $(plots_f)
	
	
# requires python 3.6.8
install: \
	install_requirements \
	install_ctapipe \
	install_aict



directories:
	mkdir -p $(prep_f)/diffuse
	mkdir -p $(prep_f)/pointlike
	mkdir -p $(prep_f)/proton
	mkdir -p $(prep_f)/diffuse_mono
	mkdir -p $(prep_f)/pointlike_mono
	mkdir -p $(prep_f)/proton_mono
	mkdir -p $(merged_f)
	mkdir -p $(models_f)
	mkdir -p $(dl2_f)
	mkdir -p $(plots_f)/test
	mkdir -p $(plots_f)/gamma
	mkdir -p $(plots_f)/test_cut
	mkdir -p $(plots_f)/gamma_cut


# python=3.6.8
.ONESHELL:
install_aict:
	cd software/aict-tools
	pip install -e .

.ONESHELL:
install_ctapipe:
	cd software/ctapipe
	pip install -e .

.ONESHELL:
install_requirements:
	cd software
	pip install -r requirements.txt


# requires the simtel files.
preprocessing:
	python $(prep_script) \
		'/net/cta-tank.app.tu-dortmund.de/POOL/projects/cta/PROD3/gammas/*run*.simtel.gz' \
		$(prep_f)/pointlike/ \
		config/preprocessing_config.yaml \
		-l config/preprocessing_logging.yaml \
		--n_jobs 8 \
		-c 12
	python $(prep_script) \
		'/net/cta-tank.app.tu-dortmund.de/POOL/projects/cta/PROD3/gammas_diffuse/*run*.simtel.gz' \
		$(prep_f)/diffuse/ \
		config/preprocessing_config.yaml \
		-l config/preprocessing_logging.yaml \
		--n_jobs 8 \
		-c 12
	python $(prep_script) \
		'/net/cta-tank.app.tu-dortmund.de/POOL/projects/cta/PROD3/protons/*run*.simtel.gz' \
		$(prep_f)/proton/ \
		config/preprocessing_config.yaml \
		-l config/preprocessing_logging.yaml \
		--n_jobs 8 \
		-c 12

preprocessing_mono:
	python $(prep_script) \
		'/net/cta-tank.app.tu-dortmund.de/POOL/projects/cta/PROD3/gammas/*run*.simtel.gz' \
		$(prep_f)/pointlike_mono/ \
		config/preprocessing_config_mono.yaml \
		-l config/preprocessing_logging.yaml \
		--n_jobs 8 \
		-c 12
	python $(prep_script) \
		'/net/cta-tank.app.tu-dortmund.de/POOL/projects/cta/PROD3/gammas_diffuse/*run*.simtel.gz' \
		$(prep_f)/diffuse_mono/ \
		config/preprocessing_config_mono.yaml \
		-l config/preprocessing_logging.yaml \
		--n_jobs 8 \
		-c 12
	python $(prep_script) \
		'/net/cta-tank.app.tu-dortmund.de/POOL/projects/cta/PROD3/protons/*run*.simtel.gz' \
		$(prep_f)/proton_mono/ \
		config/preprocessing_config_mono.yaml \
		-l config/preprocessing_logging.yaml \
		--n_jobs 8 \
		-c 12


# 299, 7, 1002
#### split these up again
merge_data:
	python $(merge_script) \
		'$(prep_f)/diffuse/*.h5' \
		$(merged_f)/diffuse_merged.hdf5 \
		-f h5py
	python $(merge_script) \
		'$(prep_f)/pointlike/*.h5' \
		$(merged_f)/gamma_merged.hdf5 \
		-f h5py
	python $(merge_script) \
		'$(prep_f)/proton/*.h5' \
		$(merged_f)/proton_merged.hdf5 \
		-f h5py
	python $(merge_script) \
		'$(prep_f)/diffuse_mono/*.h5' \
		$(merged_f)/diffuse_mono_merged.hdf5 \
		-f h5py
	python $(merge_script) \
		'$(prep_f)/pointlike_mono/*.h5' \
		$(merged_f)/gamma_mono_merged.hdf5 \
		-f h5py
	python $(merge_script) \
		'$(prep_f)/proton_mono/*.h5' \
		$(merged_f)/proton_mono_merged.hdf5 \
		-f h5py

# we need specific mono data!!!!

# 		-f 0.2 -n train_energy \
		-f 0.2 -n train_array_energy \

# 		-f 0.2 -n train_energy \
		-f 0.2 -n train_array_energy \

split_data:
	aict_split_data \
		$(merged_f)/diffuse_merged.hdf5 \
		$(merged_f)/diffuse \
		-f 0.7 -n test \
		-f 0.3 -n train \
		-t cta \
		-v
	aict_split_data \
		$(merged_f)/gamma_merged.hdf5 \
		$(merged_f)/gamma \
		-f 1 -n test \
		-t cta \
		-v
	aict_split_data \
		$(merged_f)/proton_merged.hdf5 \
		$(merged_f)/proton \
		-f 0.3 -n train_separation \
		-t cta \
		-v
	aict_split_data \
		$(merged_f)/diffuse_mono_merged.hdf5 \
		$(merged_f)/diffuse_mono \
		-f 0.7 -n test \
		-f 0.3 -n train \
		-t cta \
		-v
	aict_split_data \
		$(merged_f)/gamma_mono_merged.hdf5 \
		$(merged_f)/gamma_mono \
		-f 1 -n test \
		-t cta \
		-v
	aict_split_data \
		$(merged_f)/proton_mono_merged.hdf5 \
		$(merged_f)/proton_mono \
		-f 0.3 -n train_separation \
		-t cta \
		-v


train_models: \
	$(models_f)/separator_model.pkl \
	$(models_f)/disp_model.pkl 


$(models_f)/separator_model.pkl $(models_f)/separator_performance.hdf5: $(tel_conf)  $(merged_f)/proton_train_separation.hdf5 $(merged_f)/diffuse_train.hdf5
	aict_train_separation_model \
		$(tel_conf) \
		$(merged_f)/diffuse_train.hdf5 \
		$(merged_f)/proton_train_separation.hdf5 \
		$(models_f)/separator_performance.hdf5 \
		$(models_f)/separator_model.pkl


$(models_f)/regressor_model.pkl $(models_f)/regressor_performance.hdf5: $(tel_conf)  $(merged_f)/diffuse_train_energy.hdf5
	aict_train_energy_regressor\
		$(tel_conf) \
		$(merged_f)/diffuse_train.hdf5 \
		$(models_f)/regressor_performance.hdf5 \
		$(models_f)/regressor_model.pkl


$(models_f)/array_regressor_model.pkl $(models_f)/array_regressor_performance.hdf5: $(array_conf)  $(merged_f)/diffuse_train_array_energy.hdf5 $(models_f)/separator_model.pkl $(models_f)/regressor_model.pkl
	aict_apply_separation_model\
		$(tel_conf) \
		$(merged_f)/diffuse_train_array_energy.hdf5 \
		$(models_f)/separator_model.pkl
	aict_apply_energy_regressor\
		$(tel_conf) \
		$(merged_f)/diffuse_train_array_energy.hdf5 \
		$(models_f)/regressor_model.pkl
	aict_train_energy_regressor\
		$(array_conf) \
		$(merged_f)/diffuse_train_array_energy.hdf5 \
		$(models_f)/array_regressor_performance.hdf5 \
		$(models_f)/array_regressor_model.pkl


$(models_f)/disp_model.pkl $(models_f)/sign_model.pkl $(models_f)/cv_disp.hdf5: $(tel_conf) $(merged_f)/diffuse_train.hdf5
	aict_train_disp_regressor \
		$(tel_conf) \
		$(merged_f)/diffuse_train.hdf5 \
		$(models_f)/cv_disp.hdf5 \
		$(models_f)/disp_model.pkl \
		$(models_f)/sign_model.pkl


$(models_f)/mono_disp_model.pkl $(models_f)/mono_sign_model.pkl $(models_f)/mono_cv_disp.hdf5: $(tel_conf_mono) $(merged_f)/diffuse_mono_train_disp.hdf5
	aict_train_disp_regressor \
		$(tel_conf_mono) \
		$(merged_f)/diffuse_mono_train_disp.hdf5 \
		$(models_f)/mono_cv_disp.hdf5 \
		$(models_f)/mono_disp_model.pkl \
		$(models_f)/mono_sign_model.pkl


#### dl2 stuff

dl2: \
    $(dl2_f)/gamma_dl2.hdf5 \
	$(dl2_f)/diffuse_test_dl2.hdf5


$(dl2_f)/diffuse_mono_test_dl2.hdf5: $(merged_f)/diffuse_mono_train_disp.hdf5 $(models_f)/mono_disp_model.pkl
	cp $(merged_f)/diffuse_mono_train_disp.hdf5 $(dl2_f)/diffuse_mono_test_dl2.hdf5
	aict_apply_disp_regressor\
		$(tel_conf_mono) \
		$(dl2_f)/diffuse_mono_test_dl2.hdf5 \
		$(models_f)/mono_disp_model.pkl \
		$(models_f)/mono_sign_model.pkl

$(dl2_f)/gamma_mono_dl2.hdf5: $(merged_f)/gamma_mono_test.hdf5 $(models_f)/mono_disp_model.pkl
	cp $(merged_f)/gamma_mono_test.hdf5 $(dl2_f)/gamma_mono_dl2.hdf5
	aict_apply_disp_regressor\
		$(tel_conf_mono) \
		$(dl2_f)/gamma_mono_dl2.hdf5 \
		$(models_f)/mono_disp_model.pkl \
		$(models_f)/mono_sign_model.pkl


$(dl2_f)/gamma_dl2.hdf5: $(merged_f)/gamma_test.hdf5 $(models_f)/separator_model.pkl $(models_f)/disp_model.pkl
	cp $(merged_f)/gamma_test.hdf5 $(dl2_f)/gamma_dl2.hdf5
	aict_apply_disp_regressor\
		$(tel_conf) \
		$(dl2_f)/gamma_dl2.hdf5 \
		$(models_f)/disp_model.pkl \
		$(models_f)/sign_model.pkl
	aict_apply_stereo_disp \
		$(tel_conf) \
		$(dl2_f)/gamma_dl2.hdf5 \
		-n 20 \
		-e 10.0
	aict_apply_separation_model \
		$(tel_conf) \
		$(dl2_f)/gamma_dl2.hdf5 \
		$(models_f)/separator_model.pkl

$(dl2_f)/diffuse_test_dl2.hdf5: $(merged_f)/diffuse_test.hdf5 $(models_f)/separator_model.pkl $(models_f)/disp_model.pkl
	cp $(merged_f)/diffuse_test.hdf5 $(dl2_f)/diffuse_test_dl2.hdf5
	aict_apply_disp_regressor\
		$(tel_conf) \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(models_f)/disp_model.pkl \
		$(models_f)/sign_model.pkl
	aict_apply_separation_model \
		$(tel_conf) \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(models_f)/separator_model.pkl


	aict_apply_stereo_disp \
		$(tel_conf) \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		-n 20 \
		-e 10.0
	
cut: $(dl2_f)/diffuse_test_dl2_hadroness_cut.hdf5 $(dl2_f)/gamma_dl2_hadroness_cut.hdf5

$(dl2_f)/diffuse_test_dl2_hadroness_cut.hdf5: $(dl2_f)/diffuse_test_dl2.hdf5 $(selection_conf)
	python software/scripts/copy_array_column_to_telescope.py \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		gamma_prediction_mean
	aict_apply_cuts \
		$(selection_conf) \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(dl2_f)/diffuse_test_dl2_hadroness_cut.hdf5

$(dl2_f)/gamma_dl2_hadroness_cut.hdf5: $(dl2_f)/gamma_dl2.hdf5 $(selection_conf)
	python software/scripts/copy_array_column_to_telescope.py \
		$(dl2_f)/gamma_dl2.hdf5 \
		gamma_prediction_mean
	aict_apply_cuts \
		$(selection_conf) \
		$(dl2_f)/gamma_dl2.hdf5 \
		$(dl2_f)/gamma_dl2_hadroness_cut.hdf5



##########################################
plots: \
    disp_evaluation \
	gh_evaluation \
	test_plots \
	gamma_plots

disp_evaluation: \
	$(plots_f)/disp_features.pdf \
	$(plots_f)/sign_features.pdf \
	$(plots_f)/disp_gamma_r2_equal_filled.pdf \
	$(plots_f)/disp_test_r2_equal_filled.pdf
	
gh_evaluation: \
    $(plots_f)/separation_features.pdf \
	$(plots_f)/cross_val_sep_perf_plot.pdf

test_plots: \
	$(plots_f)/test/tel_vs_energy.pdf \
	$(plots_f)/test/multiplicity.pdf \
	$(plots_f)/test/effective_area_prep.pdf 

gamma_plots: \
	$(plots_f)/gamma/tel_vs_energy.pdf \
	$(plots_f)/gamma/multiplicity.pdf \
	$(plots_f)/gamma/effective_area_prep.pdf 


cut_plots: \
	$(plots_f)/test_cut/tel_vs_energy.pdf \
	$(plots_f)/test_cut/multiplicity.pdf \
	$(plots_f)/gamma_cut/tel_vs_energy.pdf \
	$(plots_f)/gamma_cut/multiplicity.pdf \
	$(plots_f)/median_all_cut_compare.pdf \
	$(plots_f)/gamma_cut/effective_area_prep.pdf

$(plots_f)/disp_features.pdf: $(models_f)/disp_model.pkl
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/plot_feature_importance.py \
		$(models_f)/disp_model.pkl \
		disp \
		$(tel_conf) \
		$(plots_f)/disp_features.pdf

$(plots_f)/sign_features.pdf: $(models_f)/sign_model.pkl
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/plot_feature_importance.py \
		$(models_f)/sign_model.pkl \
		disp \
		$(tel_conf) \
		$(plots_f)/sign_features.pdf

$(plots_f)/disp_features_mono_lst.pdf: $(models_f)/mono_disp_model.pkl
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/plot_feature_importance.py \
		$(models_f)/mono_disp_model.pkl \
		disp \
		$(tel_conf_mono) \
		$(plots_f)/disp_features_mono_lst.pdf


$(dl2_f)/disp_performance_gamma.hdf: $(models_f)/disp_model.pkl
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/disp_performance.py \
		$(dl2_f)/gamma_dl2.hdf5 \
		$(models_f)/disp_model.pkl \
		$(models_f)/sign_model.pkl \
		$(tel_conf) \
		$(dl2_f)/disp_performance_gamma.hdf
		
$(plots_f)/disp_gamma_r2_equal_filled.pdf: $(dl2_f)/disp_performance_gamma.hdf
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/plot_disp_performance.py \
		$(dl2_f)/disp_performance_gamma.hdf \
		$(plots_f)/disp_gamma

$(dl2_f)/disp_performance_test.hdf: $(models_f)/disp_model.pkl
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/disp_performance.py \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(models_f)/disp_model.pkl \
		$(models_f)/sign_model.pkl \
		$(tel_conf) \
		$(dl2_f)/disp_performance_test.hdf
		
$(plots_f)/disp_test_r2_equal_filled.pdf: $(dl2_f)/disp_performance_test.hdf
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/plot_disp_performance.py \
		$(dl2_f)/disp_performance_test.hdf \
		$(plots_f)/disp_test

$(dl2_f)/disp_performance_gamma_mono.hdf: $(models_f)/mono_disp_model.pkl
	TEXINPUTS=$$(pwd): \$(plots_f)/median_all_cut_compare.pdf
		$(dl2_f)/disp_performance_gamma_mono.hdf
		
$(plots_f)/disp_gamma_mono_lst_r2_equal_filled.pdf: $(dl2_f)/disp_performance_gamma_mono.hdf
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/plot_disp_performance.py \
		$(dl2_f)/disp_performance_gamma_mono.hdf \
		$(plots_f)/disp_gamma_mono_lst

$(dl2_f)/disp_performance_test_mono.hdf: $(models_f)/mono_disp_model.pkl 
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/disp_performance.py \
		$(dl2_f)/diffuse_mono_test_dl2.hdf5 \
		$(models_f)/mono_disp_model.pkl \
		$(models_f)/mono_sign_model.pkl \
		$(tel_conf_mono) \
		$(dl2_f)/disp_performance_test_mono.hdf
		
$(plots_f)/disp_test_mono_lst_r2_equal_filled.pdf: $(dl2_f)/disp_performance_test_mono.hdf
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/plot_disp_performance.py \
		$(dl2_f)/disp_performance_test_mono.hdf \
		$(plots_f)/disp_test_mono_lst

$(plots_f)/cross_val_reg_perf_plot.pdf: $(tel_conf) $(models_f)/regressor_performance.hdf5 $(models_f)/regressor_model.pkl
	TEXINPUTS=$$(pwd): \
	aict_plot_regressor_performance \
		$(tel_conf) \
		$(models_f)/regressor_performance.hdf5 \
		$(models_f)/regressor_model.pkl \
		-o $(plots_f)/cross_val_reg_perf_plot.pdf


$(plots_f)/cross_val_array_reg_perf_plot.pdf: $(array_conf) $(models_f)/array_regressor_performance.hdf5 $(models_f)/array_regressor_model.pkl
	TEXINPUTS=$$(pwd): \
	aict_plot_regressor_performance \
		$(array_conf) \
		$(models_f)/array_regressor_performance.hdf5 \
		$(models_f)/array_regressor_model.pkl \
		-o $(plots_f)/cross_val_array_reg_perf_plot.pdf


$(plots_f)/cross_val_sep_perf_plot.pdf: $(tel_conf) $(models_f)/separator_performance.hdf5 $(models_f)/separator_model.pkl
	TEXINPUTS=$$(pwd): \
	aict_plot_separator_performance \
		$(tel_conf) \
		$(models_f)/separator_performance.hdf5 \
		$(models_f)/separator_model.pkl \
		-o $(plots_f)/cross_val_sep_perf_plot.pdf

$(plots_f)/separation_features.pdf: $(models_f)/separator_model.pkl
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/plot_feature_importance.py \
		$(models_f)/separator_model.pkl \
		separator \
		$(tel_conf) \
		$(plots_f)/separation_features.pdf

$(plots_f)/gamma/tel_vs_energy.pdf: $(dl2_f)/gamma_dl2.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/mono_resolution_plots.py \
		$(dl2_f)/gamma_dl2.hdf5 \
		$(tel_conf) \
		$(plots_f)/gamma

$(plots_f)/gamma_mono_lst/tel_vs_energy.pdf: $(dl2_f)/gamma_mono_dl2.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/mono_resolution_plots.py \
		$(dl2_f)/gamma_mono_dl2.hdf5 \
		$(plots_f)/gamma_mono_lst

$(plots_f)/gamma/multiplicity.pdf: $(dl2_f)/gamma_dl2.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/stereo_resolution_plots.py \
		$(dl2_f)/gamma_dl2.hdf5 \
		$(plots_f)/gamma

$(plots_f)/gamma/effective_area_prep.pdf: $(dl2_f)/gamma_dl2.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/plot_effective_area.py \
		$(dl2_f)/gamma_dl2.hdf5 \
		$(plots_f)/gamma/effective_area_prep.pdf

$(plots_f)/gamma/effective_area_prep_signs.pdf: $(dl2_f)/gamma_dl2.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/plot_effective_area.py \
		$(dl2_f)/gamma_dl2.hdf5 \
		$(plots_f)/gamma/effective_area_prep_signs.pdf \
		-correct_signs

$(plots_f)/gamma_cut/tel_vs_energy.pdf: $(dl2_f)/gamma_dl2_hadroness_cut.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/mono_resolution_plots.py \
		$(dl2_f)/gamma_dl2_hadroness_cut.hdf5 \
		$(tel_conf) \
		$(plots_f)/gamma_cut

$(plots_f)/gamma_cut/multiplicity.pdf: $(dl2_f)/gamma_dl2_hadroness_cut.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/stereo_resolution_plots.py \
		$(dl2_f)/gamma_dl2_hadroness_cut.hdf5 \
		$(plots_f)/gamma_cut

$(plots_f)/gamma_cut/effective_area_prep.pdf: $(dl2_f)/gamma_dl2_hadroness_cut.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/plot_effective_area.py \
		$(dl2_f)/gamma_dl2_hadroness_cut.hdf5 \
		$(plots_f)/gamma_cut/effective_area_prep.pdf 

$(plots_f)/test/tel_vs_energy.pdf: $(dl2_f)/diffuse_test_dl2.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/mono_resolution_plots.py \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(tel_conf) \
		$(plots_f)/test

$(plots_f)/test/multiplicity.pdf: $(dl2_f)/diffuse_test_dl2.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/stereo_resolution_plots.py \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(plots_f)/test

$(plots_f)/test/effective_area_prep.pdf: $(dl2_f)/diffuse_test_dl2.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/plot_effective_area.py \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(plots_f)/test/effective_area_prep.pdf

$(plots_f)/test/effective_area_prep_signs.pdf: $(dl2_f)/diffuse_test_dl2.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/plot_effective_area.py \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(plots_f)/test/effective_area_prep_signs.pdf \
		-correct_signs

$(plots_f)/test_cut/tel_vs_energy.pdf: $(dl2_f)/diffuse_test_dl2_hadroness_cut.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/mono_resolution_plots.py \
		$(dl2_f)/diffuse_test_dl2_hadroness_cut.hdf5 \
		$(tel_conf) \
		$(plots_f)/test_cut

$(plots_f)/test_cut/multiplicity.pdf: $(dl2_f)/diffuse_test_dl2_hadroness_cut.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/stereo_resolution_plots.py \
		$(dl2_f)/diffuse_test_dl2_hadroness_cut.hdf5 \
		$(plots_f)/test_cut

$(plots_f)/test_mono_lst/tel_vs_energy.pdf: $(dl2_f)/diffuse_mono_test_dl2.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/mono_resolution_plots.py \
		$(dl2_f)/diffuse_mono_test_dl2.hdf5 \
		$(plots_f)/test_mono_lst

$(plots_f)/median_all_cut_compare.pdf: $(dl2_f)/gamma_dl2.hdf5 $(dl2_f)/gamma_dl2_hadroness_cut.hdf5
	TEXINPUTS=$$(pwd): \
	python $(scripts_f)/cut_compare_plots.py \
		$(dl2_f)/gamma_dl2.hdf5 \
		$(dl2_f)/gamma_dl2_hadroness_cut.hdf5 \
		$(plots_f)
