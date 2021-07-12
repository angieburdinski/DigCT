# Digital Contact Tracing

This package provides the simulation, analysis, and figure code for
the manuscript "Digital contact tracing contributes little to COVID-19
outbreak containment" by A. Burdinski, D. Brockmann, and B. F. Maier.

## Prerequisites

The analysis code was used and tested for Python 3.8 on CentOS, Ubuntu, and MacOS.
In order to run code in this collection, first install the requirements:

```bash
pip install -r requirements.txt
```

Models are implemented using [epipack](github.com/benmaier/epipack). To run
large-scale simulations, we use [qsuite](github.com/benmaier/qsuite), a CLI
to facilitate simulations on HPC clusters. `qsuite` will be installed when
dependencies are installed from `requirements.txt`.

## Main model

The main model, including an example configuration,
can be found in directory `main_model/`.
To run the simulation, do

```bash
cd main_model/
qsuite local
```

## Analyses 

Almost all simulations and analyses performed in the paper
can be found in `analysis_collection/tracing_sim/`.

All extracted (summarized) data can be found in
`analysis_collection/data_new.json`.

Code to produce the figures in the main text from distilled analysis
results and analyses for the locally clustered network with
exponential degree distribution can be found in
`figures_main_text_and_new_network_model/`.

Code for plots in the SI can be found in
`analysis_collection/tools.py` except for Fig. S7-S8-- those can
be found in the respective directories 
`analysis_collection/tracing_sim/results_deleting_edges_*`
and `analysis_collection/tracing_sim/results_toy_model/`.

In order to replicate the simulations, change to the directory containing the
respective analysis and run `qsuite local`, e.g. 

```bash
cd analysis_collection/tracing_sim/results_exponential_DF_NMEAS_100_ONLYSAVETIME_False/
qsuite local
```

An illustration to justify the choice of `beta = 10^(-6)` as the long range
redistribution parameter beta can be found by running
`analysis_collection/small_world_parameter.py`.
