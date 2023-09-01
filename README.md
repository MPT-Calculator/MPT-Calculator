# MPT-Calculator

MPT-Calculator is a series of python scripts which calls the NGSolve [1,2,3] high order finite element method (FEM) library

https://ngsolve.org

for computing the magnetic polarizability tensor (MPT) for object characterisation in metal detection. In the case of frequency sweeps, this is accelerated by the Proper Orthogonal Decomposition (POD) technique. It is designed as an educational and research tool for engineers, mathematicians and physicists working both academia and industry and it is hoped those interested in characterising conducting permeable objects will find it useful.

The MPT characterises the shape, conductivity, permeability of conducting permeable object, is frequency dependent [4] and is independent of the object?s position. The rank 2 MPT is symmetric and has at most 6 independent complex coefficients. However, for objects with mirror or rotational symmetries the number of independent coefficients is smaller [4,5].

MPT-Calculator computes the MPT using a range of different numerical schemes

A hp FEM discretisation of the transmission problems using NGSolve to compute MPT for a single frequency.

A hp FEM discretisation of the transmission problems using NGSolve for performing the computation of the MPT over a range of frequencies.

A Proper Orthogonal Decomposition (POD) reduced order model, which greatly accelerates the computation of the full order model in 2. for computing the MPT over a range of frequencies.

The technical details of the implementation will be described in [6].

Plots of the computed tensor coefficients as a function of frequency are created and the output data and plots are automatically stored so that they can be recreated, if desired.

# To get started

Please open the set of Jupyter Tutorials available in the Tutorial folder starting with Tutorials/a_Installation.ipynb, which describes the installation and setup of MPT-Calculator.


# Referencing

If you use the tool, please refer to it in your work by citing the references

[4] P. D. Ledger and W. R. B. Lionheart, The spectral properties of the magnetic polarizability tensor for metallic object characterisation, Math Meth Appl Sci., 2019, doi:10.1002/mma.5830

[5] P. D. Ledger and W. R. B. Lionheart, An explicit formula for the magnetic polarizability tensor for object characterization, IEEE Trans Geosci Remote Sens., 56(6), 3520-3533, 2018.

[6] B. A. Wilson and P. D. Ledger, Efficient computation of the magnetic polarizabiltiy tensor spectral signature using pod. International Journal for Numerical Methods in Engineering 122, 1940-1963, 2021.

[7] P.D. Ledger, B.A. Wilson, A.A.S. Amad, W.R.B. Lionheart, Identification of meallic objects using spectral MPT signatures: Object characterisation and invariants. International Journal for Numerical Methods in Engineering. 122(15), 2021. https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.6688

[8] J. Elgy, P.D. Ledger, J.L. Davidson, T. Ozdeger and A.J. Peyton, Computations and measurement of the magnetic polarizability tensor characterisation of highly conducting and magnetic objects. Engineering Computations, 2023, doi:10.1108/EC-11-2022-0688.

as well as those of NGSolve:

[1] J. Schoberl, C++11 Implementation of Finite Elements in NGSolve, ASC Report 30/2014, Institute for Analysis and Scientific Computing, Vienna University of Technology, 2014.

[2] S. Zaglmayr, High Order Finite Elements for Electromagnetic Field Computation, PhD Thesis, Johannes Kepler University Linz, 2006

[3] J. Schoberl, NETGEN - An advancing front 2D/3D-mesh generator based on abstract rules, Computing and Visualization in Science, 1(1), 41-52, 1997.
