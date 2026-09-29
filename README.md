# LPC3D

A code to do mesoscopic simulations of ions diffusing in carbon particles and of full supercapacitors.

This code is developed in the context of the MultiXscale project.

This code was written by El Hassane Lahrar and Céline Merlet, with contributions from Rudolf Weeber.

Here, a manual and example input files are provided.

Compared to the previous version of the program, written in C and serial (https://github.com/cmerlet/LPC3D-C-serial), this code is written using pystencils (https://pypi.org/project/pystencils/), is parallel and can use CPU and GPU.

## Multiscale coupling with LAMMPS

The `LAMMPS-LPC3D/` folder contains a coupled version of the code, in which LPC3D drives an ensemble of independent LAMMPS simulations, one for each pore size. The pore-resolved densities obtained from molecular dynamics are transferred on the fly to the lattice gas model. The coupling is implemented in Python with mpi4py and the LAMMPS Python interface, and can be run on CPU or GPU (LAMMPS with Kokkos). Examples for both modes are provided in `LAMMPS-LPC3D/Examples/`, and the installation, input format and usage are described in the manual.

## Publications using LPC3D

If you use LPC3D in your work, please cite the relevant references below.

1. E. H. Lahrar, M. H. Mohamed, M. Salanne, G. Jeanmairet, C. van Leeuwen and C. Merlet, "Coupling LAMMPS with LPC3D for Mesoscopic Simulations of Electrolytes in Porous Carbons: A Python-MPI Workflow for CPU and GPU Supercomputers", *Procedia Computer Science*, 286, 58-65 (2026), Proceedings of the fourth EuroHPC User Days. https://doi.org/10.1016/j.procs.2026.08.019
   *(Reference for the LAMMPS-LPC3D coupling.)*

2. E. H. Lahrar, M. Salanne, R. Weeber and C. Merlet, "LPC3D: An Enhanced Parallel Software for Large-Scale Simulation of Adsorption in Porous Carbons and Supercapacitors", arXiv:2603.22553 (2026). https://arxiv.org/abs/2603.22553
   *(Reference for the parallel PyStencils implementation.)*

3. E. H. Lahrar and C. Merlet, "Investigating the effect of particle size distribution and complex exchange dynamics on NMR spectra of ions diffusing in disordered porous carbons through a mesoscopic model", *Faraday Discussions*, 255, 355-369 (2025). https://doi.org/10.1039/d4fd00082j

## Acknowledgements

This project has received funding from the European Union, the European High Performance Computing Joint Undertaking (JU) and countries participating in the MultiXscale project under grant agreement No 101093169.

