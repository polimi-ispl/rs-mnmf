# Ray-Space-Based Multichannel Nonnegative Matrix Factorization 

## About
Matlab implementation of the Ray-Space-Based Multichannel Nonegative Matrix Factorization (RS-MCNMF) for audio source separation.
A blind source separation is performed adopting the MCNMF algorithm to the Ray Space data. 

## Contents

```
.
├── LICENSE
├── README.md
├── code
│   ├── lib
│   ├── rayspacenmf.m
│   ├── rsmcnmf_example.m
├── data
```

- `code`: folder with the source code.
    - `lib`: folder with utilities for the BS-MCNMF, MCNMF evaluation and other.
    - `rayspacenmf.m`: MATLAB function for the RS-MCNMF.
    - `rsmcnmf_example.m`: example script for RS-MCNMF source separation and comparison among the techniques
- `data`: folder with sample data for `rsmcnmf_example.m`