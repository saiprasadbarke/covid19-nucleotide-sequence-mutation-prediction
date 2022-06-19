SARS-Cov-2 genome evolution prediction using Encoder-Decoder Architecture.
==============================

Using an  Encoder- Decoder architecture with Bahdanau Attention to predict the mutations between clades in the Spike Nucleotide sequences of the SARS-Cov-2 virus genome.

Getting Started
------------
```angular2html
1. cd CondaRequirements/
2. conda env create -f environment.yml
3. conda activate CNSMP
```

Raw Dataset 
------------
The raw dataset containing the genomes can be found on the GISAID website here: 
https://www.epicov.org/epi3/

Project Organization
------------

    ├── README.md          
    ├── data
    │   ├── clades.tabular        <- Clades assigned to sequences usingthe  nextclade tool.
    │   ├── sequences.fasta       <- Sequence Data without any ambiguious characters
    │   └── merged.json           <- Merged data linking sequences and clades       
    ├── notebooks                 <- Jupyter notebooks. 
    ├── reports                   <- Generated analysis json and png plots.
    │   ├── stats                 <- Json files of the plot data
    │   └── plots                 <- Plots
    ├── CondaRequirements       
    │   └── environment.yml       <- The requirements file for reproducing the analysis environment
    └── src                       <- Source code for use in this project.
        │
        ├── data_operations       <- Scripts to clean and generate data
        ├── helpers               <- Miscellaneous helper functions  
        ├── inference             <- Inference script
        ├── metrics               <- Functions to compute prediction metrics
        ├── model_components      <- Components of the neural network
        ├── model_training        <- Scripts to train the model
        ├── scripts               <- Entry scripts
        ├── settings              <- Settings and constants
        └── visualization         <- Scripts to create visualizations of data and results
------------
