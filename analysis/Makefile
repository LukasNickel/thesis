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
selection_conf = config/aict_selection.yaml


.PHONY: merge_data split_data analysis plots all clean

from_scratch: directories install preprocessing analysis thesis

all: analysis plots thesis


analysis: merge_data split_data train_models plots

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
	mkdir -p $(merged_f)
	mkdir -p $(models_f)
	mkdir -p $(dl2_f)
	mkdir -p $(plots_f)/test
	mkdir -p $(plots_f)/gamma


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
		'/net/cta-tank.app.tu-dortmund.de/POOL/projects/cta/PROD3/gammas/*run79*.simtel.gz' \
		$(prep_f)/pointlike/ \
		config/preprocessing_config.yaml \
		-l config/preprocessing_logging.yaml \
		--n_jobs 8 \
		-c 12
	python $(prep_script) \
		'/net/cta-tank.app.tu-dortmund.de/POOL/projects/cta/PROD3/gammas_diffuse/*run299*.simtel.gz' \
		$(prep_f)/diffuse/ \
		config/preprocessing_config.yaml \
		-l config/preprocessing_logging.yaml \
		--n_jobs 8 \
		-c 12
	python $(prep_script) \
		'/net/cta-tank.app.tu-dortmund.de/POOL/projects/cta/PROD3/protons/*run1002*.simtel.gz' \
		$(prep_f)/proton/ \
		config/preprocessing_config.yaml \
		-l config/preprocessing_logging.yaml \
		--n_jobs 8 \
		-c 12




#### split these up again
merge_data:
	python $(merge_script) \
		'$(prep_f)/diffuse/*run*.h5' \
		$(merged_f)/diffuse_merged.hdf5 \
		-f h5py
	python $(merge_script) \
		'$(prep_f)/pointlike/*run*.h5' \
		$(merged_f)/gamma_merged.hdf5 \
		-f h5py
	python $(merge_script) \
		'$(prep_f)/proton/*run*.h5' \
		$(merged_f)/proton_merged.hdf5 \
		-f h5py



split_data:
	aict_split_data \
		$(merged_f)/diffuse_merged.hdf5 \
		$(merged_f)/diffuse \
		-f 0.2 -n test \
		-f 0.2 -n train_disp \
		-f 0.2 -n train_energy \
		-f 0.2 -n train_array_energy \
		-f 0.2 -n train_separation \
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
		-f 1 -n train_separation \
		-t cta \
		-v

train_models: \
	$(models_f)/separator_model.pkl \
	$(models_f)/regressor_model.pkl \
	$(models_f)/disp_model.pkl \
	$(models_f)/array_regressor_model.pkl


$(models_f)/separator_model.pkl $(models_f)/separator_performance.hdf5: $(tel_conf)  $(merged_f)/proton_train_separation.hdf5 $(merged_f)/diffuse_train_separation.hdf5
	aict_train_separation_model \
		$(tel_conf) \
		$(merged_f)/diffuse_train_separation.hdf5 \
		$(merged_f)/proton_train_separation.hdf5 \
		$(models_f)/separator_performance.hdf5 \
		$(models_f)/separator_model.pkl


$(models_f)/regressor_model.pkl $(models_f)/regressor_performance.hdf5: $(tel_conf)  $(merged_f)/diffuse_train_energy.hdf5
	aict_train_energy_regressor\
		$(tel_conf) \
		$(merged_f)/diffuse_train_energy.hdf5 \
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


$(models_f)/disp_model.pkl $(models_f)/sign_model.pkl $(models_f)/cv_disp.hdf5: $(tel_conf) $(merged_f)/diffuse_train_disp.hdf5
	aict_train_disp_regressor \
		$(tel_conf) \
		$(merged_f)/diffuse_train_disp.hdf5 \
		$(models_f)/cv_disp.hdf5 \
		$(models_f)/disp_model.pkl \
		$(models_f)/sign_model.pkl


#### dl2 stuff 
$(dl2_f)/gamma_dl2.hdf5: $(merged_f)/gamma_test.hdf5 $(models_f)/separator_model.pkl $(models_f)/regressor_model.pkl $(models_f)/disp_model.pkl
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

$(dl2_f)/diffuse_test_dl2.hdf5: $(merged_f)/diffuse_test.hdf5 $(models_f)/separator_model.pkl $(models_f)/regressor_model.pkl $(models_f)/disp_model.pkl $(models_f)/array_regressor_model.pkl
	cp $(merged_f)/diffuse_test.hdf5 $(dl2_f)/diffuse_test_dl2.hdf5
	aict_apply_disp_regressor\
		$(tel_conf) \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(models_f)/disp_model.pkl \
		$(models_f)/sign_model.pkl
	aict_apply_stereo_disp \
		$(tel_conf) \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		-n 20 \
		-e 10.0
	aict_apply_separation_model \
		$(tel_conf) \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(models_f)/separator_model.pkl
	aict_apply_energy_regressor \
		$(tel_conf) \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(models_f)/regressor_model.pkl
	aict_apply_energy_regressor \
		$(array_conf) \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(models_f)/array_regressor_model.pkl


cut: $(dl2_f)/diffuse_test_dl2_hadroness_cut.hdf5

$(dl2_f)/diffuse_test_dl2_hadroness_cut.hdf5: $(dl2_f)/diffuse_test_dl2.hdf5 $(selection_conf)
	aict_apply_cuts \
		$(selection_conf) \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(dl2_f)/diffuse_test_dl2_hadroness_cut.hdf5 \
		--key array_events \
		--chunksize=10000



plots: \
	$(plots_f)/disp_features.pdf \
	$(plots_f)/reg_perf_plot.pdf \
	$(plots_f)/array_reg_perf_plot.pdf \
	$(plots_f)/sep_perf_plot.pdf \
	$(plots_f)/test/tel_vs_energy.pdf \
	$(plots_f)/gamma/tel_vs_energy.pdf



$(plots_f)/disp_features.pdf: $(models_f)/disp_model.pkl
	python $(scripts_f)/plot_feature_importance.py \
		$(models_f)/disp_model.pkl \
		$(tel_conf) \
		$(plots_f)/disp_features.pdf

$(plots_f)/gamma/tel_vs_energy.pdf: $(dl2_f)/gamma_dl2.hdf5
	python $(scripts_f)/resolution_plots.py \
		$(dl2_f)/gamma_dl2.hdf5 \
		$(plots_f)/gamma \
		$(tel_conf)


$(plots_f)/test/tel_vs_energy.pdf: $(dl2_f)/diffuse_test_dl2.hdf5
	python $(scripts_f)/resolution_plots.py \
		$(dl2_f)/diffuse_test_dl2.hdf5 \
		$(plots_f)/test \
		$(tel_conf)


$(plots_f)/reg_perf_plot.pdf: $(tel_conf) $(models_f)/regressor_performance.hdf5 $(models_f)/regressor_model.pkl
	aict_plot_regressor_performance \
		$(tel_conf) \
		$(models_f)/regressor_performance.hdf5 \
		$(models_f)/regressor_model.pkl \
		-o $(plots_f)/reg_perf_plot.pdf


$(plots_f)/array_reg_perf_plot.pdf: $(array_conf) $(models_f)/array_regressor_performance.hdf5 $(models_f)/array_regressor_model.pkl
	aict_plot_regressor_performance \
		$(array_conf) \
		$(models_f)/array_regressor_performance.hdf5 \
		$(models_f)/array_regressor_model.pkl \
		-o $(plots_f)/array_reg_perf_plot.pdf


$(plots_f)/sep_perf_plot.pdf: $(tel_conf) $(models_f)/separator_performance.hdf5 $(models_f)/separator_model.pkl
	aict_plot_separator_performance \
		$(tel_conf) \
		$(models_f)/separator_performance.hdf5 \
		$(models_f)/separator_model.pkl \
		-o $(plots_f)/sep_perf_plot.pdf

