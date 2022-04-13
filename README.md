SARS-Cov-2 genome evolution prediction using Encoder-Decoder Architecture.
==============================

Using an Attention Encoder- Decoder architecture to predict the mutations between clades in Nucleotide sequences of the SARS-Cov-2 virus genome.

Getting Started
------------
```angular2html
1. cd CondaRequirements/
2. conda env create -f environment.yml
3. conda activate CNSMP
```

Raw Dataset 
------------
The raw dataset containing the genomes can be found on the NCBI website here: 
https://www.ncbi.nlm.nih.gov/datasets/coronavirus/genomes/

Project Organization
------------

    ├── README.md          
    ├── data
    │   ├── 00unfiltered        <- Unfiltered data from the NCBI website + their clades from the nextclade tool.
    │   ├── 01cleaned           <- Data without any ambiguious characters + their clades from the nextclade tool
    │   ├── 02merged            <- Sequences merged with the relevant clades (Obtained from 01cleaned)
    │   ├── 03paired       
    │   └── 04permuted 
    │       
    │
    ├── models                  <- Trained and serialized models, model predictions, or model summaries
    │   
    ├── notebooks               <- Jupyter notebooks. 
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── plots               <- Generated graphics and figures to be used in reporting
    │
    ├── CondaRequirements       
    │   └── environment.yml     <- The requirements file for reproducing the analysis environment
    │
    └── src                     <- Source code for use in this project.
        │
        ├── data_operations     <- Scripts to clean and generate data
        │   ├── 
        │   └── 
        │
        │
        ├── models              <- 
        │   ├── 
        │
        └── visualization       <- Scripts to create exploratory and results oriented visualizations
            └── 
    


--------
