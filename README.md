#MPHvect

This repository contains a Python implementation of the method described in:

Peter Bubenik & Zachariah Ross, "A Schauder Basis for Multiparameter Persistence", arXiv:2510.10347


This method uses iteratively refined triangulations to vectorize signed barcodes from multiparameter persistence, as well as persistence diagrams from 1-parameter persistence. 

Time: While Numba is not a necessary requirement for this, it is recommended that you have numba installed when running this code, as it drastically decreases run times. 


Optional dependencies (for examples): 
The example scripts that demonstrate MPHvect with persistence diagrams and signed barcodes use the packages `multipers`, `ripser`, and `persim` . 
Install them with:
    pip install multipers
    pip install ripser
    pip install persim


---

## Installation

Clone the repository:

    git clone https://github.com/zachtross/MPHvect.git
    cd MPHvect

Install dependencies:

    pip install -r requirements.txt

---


