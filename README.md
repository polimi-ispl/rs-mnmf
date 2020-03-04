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
    - `lib`: folder with utilities for the BS-MCNMF, MCNMF, BSS evaluation and more.
    - `rayspacenmf.m`: MATLAB function for the RS-MCNMF.
    - `rsmcnmf_example.m`: example script for RS-MCNMF source separation and comparison among the techniques
- `data`: folder with sample data for `rsmcnmf_example.m`

## Usage

Clone or download the repository and run `rsmcnmf_example.m` to see how to use the function `rayspacenmf.m`.

## References

The RS-MCNMF for audio source separation was originally proposed in:
* M. Pezzoli, M. Cobos, F. Antonacci, A. Sarti, "Ray-Space-Based Multichannel Nonnegative Matrix Factorization for Audio Source Separation" 

However the following articles are also important for understanding the technique:

* A. Ozerov and C. Févotte, "Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation," IEEE Transaction on Audio, Speech, and Language, Processing, vol. 18, no. 3, pp. 550–563, 2010.
* S. Lee, S. H. Park and K. Sung, "Beamspace-Domain Multichannel Nonnegative Matrix Factorization for Audio Source Separation," in IEEE Signal Processing Letters, vol. 19, no. 1, pp. 43-46, Jan. 2012.
* L. Bianchi, F. Antonacci, A. Sarti and S. Tubaro, "The Ray Space Transform: A New Framework for Wave Field Processing," in IEEE Transactions on Signal Processing, vol. 64, no. 21, pp. 5696-5706, 1 Nov.1, 2016.

## See also
[ISPL website](http://ispl.deib.polimi.it)
