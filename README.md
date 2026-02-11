The file RecognizeRegularSingularMahlerSystem.py contains an implementation of the main algorithm from the paper C. Faverjon, M. Poulet, An algorithm to recognize regular singular Mahler systems, Math.
Comp. 91 (2022), 2905–2928.
Given a linear p-Mahler system, it returns whether the system is regular singular at 0 and if so, it returns the constant matrix it is Puiseux-equivalent to and as many coefficients as wanted of the associated gauge transform.

The file Algorithm_FundamentalMatrixSolutions_MahlerSystems.py contains an implementation in Python of Algorithm 1 from the paper "Computing basis of solutions of any Mahler equation" from C. Faverjon and M. Poulet (preprint, 2025).
Given a linear p-Mahler system with matrix A(z), it returns an integer d, a matrix Theta, with constant non-zero determinant and coefficients in K[z^{-1}] as well as a truncation of a matrix P with Laurent series coefficients such that P(z^p)Theta=A(z^d)P(z).

The PDF file contains some examples.
