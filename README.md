# Ray-Space-Based Multichannel Nonnegative Matrix Factorization 

## About
Matlab implementation of the Ray-Space-Based Multichannel Nonegative Matrix Factorization (RS-MNMF) for audio source separation.
A blind source separation is performed adopting the MNMF algorithm to the Ray Space data. 

## Abstract
Nonnegative matrix factorization (NMF) has been traditionally considered a promising approach for audio source separation. 
While standard NMF is only suited for single-channel mixtures, extensions to consider multi-channel data have been also proposed.
Among the most popular alternatives, multichannel NMF (MNMF) and further derivations based on constrained spatial covariance models have been successfully employed to separate multi-microphone convolutive mixtures. 
This letter proposes a MNMF extension by considering a mixture model with Ray-Space-transformed signals, where magnitude data successfully encodes source locations as frequency-independent linear patterns. 
We show that the MNMF algorithm can be seamlessly adapted to consider Ray-Space-transformed data, providing competitive results with recent state-of-the-art MNMF algorithms in a number of configurations using real recordings.

## Contents

```
.
├── LICENSE
├── README.md
├── code
│   ├── lib
│   ├── rayspacenmf.m
│   ├── rsmnmf_example.m
├── data
```

- `code`: folder with the source code.
    - `lib`: folder with utilities for BSS evaluation and more.
    - `rayspacenmf.m`: MATLAB function for the RS-MNMF.
    - `rsmnmf_example.m`: example script for RS-MNMF source separation.
- `data`: folder with the RIR dataset and source signals adopted in the SPL publication.

## Usage

Clone or download the repository and run `rsmnmf_example.m` to see how to use the function `rayspacenmf.m`.

## References

The RS-MNMF for audio source separation was originally proposed in:
* M. Pezzoli, J. J. Carabias-Orti, M. Cobos, F. Antonacci, A. Sarti, "Ray-Space-Based Multichannel Nonnegative Matrix Factorization for Audio Source Separation",  IEEE Signal Processing Letters (2021), doi: 10.1109/LSP.2021.3055463 

However the following articles are also important for understanding the technique:

* A. Ozerov and C. Févotte, "Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation," IEEE Transaction on Audio, Speech, and Language, Processing, vol. 18, no. 3, pp. 550–563, 2010.
* S. Lee, S. H. Park and K. Sung, "Beamspace-Domain Multichannel Nonnegative Matrix Factorization for Audio Source Separation," in IEEE Signal Processing Letters, vol. 19, no. 1, pp. 43-46, Jan. 2012.
* L. Bianchi, F. Antonacci, A. Sarti and S. Tubaro, "The Ray Space Transform: A New Framework for Wave Field Processing," in IEEE Transactions on Signal Processing, vol. 64, no. 21, pp. 5696-5706, 1 Nov.1, 2016.

## See also
[ISPL website](http://ispl.deib.polimi.it), [SPAT  website](https://spat.blogs.uv.es)
