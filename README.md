# Digital Contact Tracing

This package provides the simulation, analysis, and figure code for the manuscript "Digital
contact tracing contributes little to COVID-19 outbreak containment" by A. Burdinski, D. Brockmann, and B. F. Maier.

All simulations (including the model) can be found in `_qsuite/tracing_sim`.

All extracted data can be found in `_qsuite/data_new.json`.

Code for plots in main text and analysis for the locally clustered network with
exponential degree distribution can be found in `main_figures_and_new_network_model/`.

Code for plots in SI can be found in `_qsuite/tools.py` except for Fig. S7-S8 those can
be found in the respective `_qsuite/results_deleting_edges...`.

Calculations for the long range redistribution parameter beta can be found in
`_qsuite/small_world_parameter.py` .
