#!/usr/bin/env python
# coding: utf-8
"""
This program is a lattice model to calculate quantities of adsorbed ions, diffusion coefficients and predict NMR spectra of ions adsorbed in 
porous carbon electrodes in supercapacitors:
 - for diffusion coefficients calculations, data is derived from molecular dynamics simulations (LAMMPS) and is used as input to determine in-pore adsorption profiles, 
   and data from experiments is used to obtain energy barriers governing transitions between lattice sites;
 - for NMR spectra calculations, the model uses chemical shifts obtained from density functional theory calculations.

To run code : 
export OMP_NUM_THREADS=<N_THREADS>
export LPC3D_NUM_GPUS=$N_GPUS # If GPU is used

srun python3 LAMMPS-LPC3D.py -i lattice_gas.inpt
Or:
mpirun -np <N_MPI_PROCS> python3 LAMMPS-LPC3D.py -i lattice_gas.inpt
"""


from pystencils.session import *
import math
import time
import sys
from numba import njit, set_num_threads, prange, typed
import os
import argparse
import subprocess
import mpi4py_lammps
from mpi4py import MPI


verif=1


def propagation(steps):
        global sumvacf, sumvacfx, sumvacfy,sumvacfz, ireal,xsumvacf,xsumvacfx,xsumvacfy,xsumvacfz,ysumvacf,ysumvacfx,ysumvacfy,ysumvacfz,zsumvacf,zsumvacfx,zsumvacfy,zsumvacfz

        for i in range(2,steps):
            if i%display_interval==0:
                print("Timestep",i)
            sync1()
            sync3()
            sync4()
            sync5()
            dh.run_kernel(kernel)
            dh.swap(probv0.name, probv0_next.name)
            dh.swap(ReG.name, ReG_next.name)
            dh.swap(ImG.name, ImG_next.name)
            dh.swap(c.name, c_next.name)
            dh.swap(locpfunc.name, locpfunc_next.name)
            dh.swap(Vmoy.name, Vmoy_next.name)

            dh.run_kernel(kernel1)
            x_arrays[ReG.name]=xp.nan_to_num(x_arrays[ReG.name], nan=0.0)
            x_arrays[ImG.name]=xp.nan_to_num(x_arrays[ImG.name], nan=0.0)
            dh.run_kernel(kernel2)
            dh.run_kernel(kernel3)
            x_arrays[probv0.name]=xp.nan_to_num(x_arrays[probv0.name], nan=0.0)
            dh.run_kernel(kernel4)
            # -------- this loop calculate quantities for each block-------
            for nb in range(1, nblocks+1):
                lattvacfx[nb, i]= xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0])
                lattvacfy[nb, i]= xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,1])
                lattvacfz[nb, i]= xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,2])
                lattvacf[nb, i] = lattvacfx[nb, i]+lattvacfy[nb, i]+lattvacfz[nb, i]

                denstot[nb, i]=xp.sum(x_arrays[c.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2])
                
                ReGtot[nb, i]=xp.sum(x_arrays[ReGs.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2])/denstot[nb, i]
                ImGtot[nb, i]=xp.sum(x_arrays[ImGs.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2])/denstot[nb, i]

                xvacfx[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0], axis=(1,2))
                xvacfy[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,1], axis=(1,2))
                xvacfz[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,2], axis=(1,2))
                
                xvacf[nb]= xvacfx[nb] + xvacfy[nb] + xvacfz[nb]
                
                yvacfx[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0], axis=(0,2))
                yvacfy[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,1], axis=(0,2))
                yvacfz[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,2], axis=(0,2))
                
                yvacf[nb] = yvacfx[nb] + yvacfy[nb] + yvacfz[nb]
                
                zvacfx[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0], axis=(0,1))
                zvacfy[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,1], axis=(0,1))
                zvacfz[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,2], axis=(0,1))

                zvacf[nb] = zvacfx[nb] + zvacfy[nb] + zvacfz[nb]

                xsumvacf[nb] += xvacf[nb] * dt / 2.0; xsumvacfx[nb] += xvacfx[nb] * dt / 2.0; xsumvacfy[nb] += xvacfy[nb] * dt / 2.0; xsumvacfz[nb] += xvacfz[nb] * dt / 2.0
                ysumvacf[nb] += yvacf[nb] * dt / 2.0; ysumvacfx[nb] += yvacfx[nb] * dt / 2.0; ysumvacfy[nb] += yvacfy[nb] * dt / 2.0; ysumvacfz[nb] += yvacfz[nb] * dt / 2.0
                zsumvacf[nb] += zvacf[nb] * dt / 2.0; zsumvacfx[nb] += zvacfx[nb] * dt / 2.0; zsumvacfy[nb] += zvacfy[nb] * dt / 2.0; zsumvacfz[nb] += zvacfz[nb] * dt / 2.0
                    
                sumvacf[nb, i]=sumvacf[nb, i-1] +  (lattvacf[nb, i]*dt/denstot[nb, i])
                sumvacfx[nb, i]=sumvacfx[nb, i-1] +  (lattvacfx[nb, i]*dt/denstot[nb, i])
                sumvacfy[nb, i]=sumvacfy[nb, i-1] +  (lattvacfy[nb, i]*dt/denstot[nb, i])
                sumvacfz[nb, i]=sumvacfz[nb, i-1] +  (lattvacfz[nb, i]*dt/denstot[nb, i])

                if i%nsample==0 :
                    ReGtotsmp[nb, ireal]=ReGtot[nb, i]
                    ImGtotsmp[nb, ireal]=ImGtot[nb, i]
                    magnitude[nb, ireal] = xp.sqrt(ReGtot[nb, i]**2 + ImGtot[nb, i]**2)
                    phase[nb, ireal] = xp.arctan2(ImGtot[nb, i], ReGtot[nb, i])
                    
            if i%nsample==0 :
                ireal+=1


def time_loop(nsteps_equ):
            global verif
            for i in range(nsteps_equ):
                sync1()                              #each timestep we synchronize densities for ghost layers
                dh.run_kernel(kernel)                #run the kernel
                dh.swap(c.name, c_next.name)         #new densities take place of old densities
        
                if xp.allclose(x_arrays[c.name][1:sn1+1,1:sn2+1,1:sn3+1], x_arrays[c_next.name][1:sn1+1,1:sn2+1,1:sn3+1],atol=1e-4): #compare the new densities with old densities
                    verif=0
                    print(f"Equilibration succeeded at {i} steps!")
                    break

def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    pore_size_list = None
    zmat_files = None
    zmat_values = None
    inp_files = None
    use_gpu = None
    ngpu = None


    if rank==0:
        start_time = time.time()

        global verif, x_arrays
        global sn1, sn2, sn3, nsteps_equ
        global periodicity, a, dwell, nsample, T, larmorfreq, calculated_spec
        global Nvalues, Ea, equilibration, electrode_nature, particles_file
        global nblocks, block_names, pore_prob_file
        global density_liquid_file, xmin, xmax, ymin, ymax, zmin, zmax
        global freq_bulk, dens_bulk, ene_bulk, elec_fil
        global ReGtot, ImGtot, ReGtotsmp, ImGtotsmp, magnitude, phase
        global sumvacf, sumvacfx, sumvacfy, sumvacfz, denstot
        global lattvacfx, lattvacfy, lattvacfz, lattvacf
        global xsumvacf, xsumvacfx, xsumvacfy, xsumvacfz
        global ysumvacf, ysumvacfx, ysumvacfy, ysumvacfz
        global zsumvacf, zsumvacfx, zsumvacfy, zsumvacfz
        global xvacf, xvacfx, xvacfy, xvacfz
        global yvacf, yvacfx, yvacfy, yvacfz
        global zvacf, zvacfx, zvacfy, zvacfz
        global sync1, sync2, sync3, sync4, sync5, sync6 
        global dh, kernel, kernel1, kernel2, kernel3, kernel4, display_interval ,xp, dt, ireal, steps, kB,v0, domain_size, dtime, Temp, kBoltz, domain_size, tau
        

        parser = argparse.ArgumentParser(
            prog='LAMMPS-LPC3D.py',
            description='Multiscale simulations involving CPU or GPU computations, reading from an input file.',
            epilog=' Usage: mpirun -np <N_MPI_PROCS> python3 LAMMPS-LPC3D.py -i lattice_gas.inpt'
        )

        parser.add_argument(
            "-i", "--input_file",
            type=str,
            required=True,
            help="Name of the input file to read."
        )

        args = parser.parse_args()

        input_file_name = args.input_file



        #--------reading input file variables----------
        try:
            with open(input_file_name, "r") as file_input:
                user_choice = file_input.readline().strip().lower()      #user choice CPU or GPU
                if user_choice == 'gpu':
                    import cupy as cp
                    xp = cp
                    target=ps.Target.GPU
                    gpu=True
                    cpu=None

                elif user_choice == 'cpu':
                    xp = np
                    gpu=None
                    cpu=True
                    
                sn1, sn2, sn3 = map(int, file_input.readline().strip().split())      # number of lattice site in x, y and z directions

                periodicity_line = file_input.readline().strip()
                periodicity_values = periodicity_line.split()
                periodicity = [val.lower() == 'true' for val in periodicity_values]  #periodicity in 3 directions

                a = xp.float32(file_input.readline().strip())              #Lattice parameter
                dwell = xp.float32(file_input.readline().strip())          #Dwell time
                nsample = int(file_input.readline().strip())               #Number of time steps between measures
                T = int(file_input.readline().strip())                     #Temperature of the simulation
                larmorfreq = int(file_input.readline().strip())            #Larmor frequency of the studied nucleus under the magnetic field considered
                Nvalues = int(file_input.readline().strip())               #Number of values for the Fourier transform 
                Ea = xp.float32(file_input.readline().strip())             #energy barrier value
                equilibration = int(file_input.readline().strip())         #equilibration process. 0 if no, 1 if yes 
                if equilibration==1:
                    nsteps_equ = int(file_input.readline().strip())
                electrode_nature= file_input.readline().strip()
                if electrode_nature=='particles':
                    particles_file=file_input.readline().strip()
                pore_prob_file = file_input.readline().strip()
                nblocks = int(file_input.readline().strip())               # number of blocks in z directions
                file_input.readline().strip()                              #empty line

                block_names = [''] * (nblocks + 1)
                pore_energy_file = [''] * (nblocks + 1)
                pore_shift_file = [''] * (nblocks + 1)
                density_liquid_file = [''] * (nblocks + 1)
            
                xmin = np.empty(nblocks+1, dtype=int)                     #create arrays for all blocks coordinates
                xmax = np.empty(nblocks+1, dtype=int)
                ymin = np.empty(nblocks+1, dtype=int)
                ymax = np.empty(nblocks+1, dtype=int)
                zmin = np.empty(nblocks+1, dtype=int)
                zmax = np.empty(nblocks+1, dtype=int)
                freq_bulk = np.empty(nblocks+1, dtype=float)
                dens_bulk = np.empty(nblocks+1, dtype=float)
                ene_bulk = np.empty(nblocks+1, dtype=float)
                elec_fil = np.empty(nblocks+1, dtype=int)

                for nb in range(1,nblocks+1):
                    block_names[nb] = file_input.readline().strip()        #name of block: electrode or bulk
                    if block_names[nb]=='electrode':
                        elec_fil[nb] = int(file_input.readline().strip())         #for electrode block, electrode is filled or not 

                    x_coords = list(map(int, file_input.readline().strip().split()))
                    y_coords = list(map(int, file_input.readline().strip().split()))
                    z_coords = list(map(int, file_input.readline().strip().split()))

                    xmin[nb] = x_coords[0]
                    xmax[nb] = x_coords[1]
                    ymin[nb] = y_coords[0]
                    ymax[nb] = y_coords[1]
                    zmin[nb] = z_coords[0]
                    zmax[nb] = z_coords[1]
                    if block_names[nb]=='electrode':
                        pore_shift_file[nb] = file_input.readline().strip()     #name of pore-frequency file 
                    elif block_names[nb]=='bulk':
                        dens_bulk[nb] = xp.float32(file_input.readline().strip())  #density of bulk 
                        freq_bulk[nb] = xp.float32(file_input.readline().strip())   #frequency of bulk 
                    file_input.readline().strip()                                #empty line

                nb_fluid = 3
                zmat_files = []
                zmat_values = []
                for _ in range(nb_fluid): 
                    zmat_line = file_input.readline().strip()
                    zmat_file, zmat_value = zmat_line.split()
                    zmat_files.append(zmat_file)
                    zmat_values.append(int(zmat_value))

                calculated_spec = int(file_input.readline().strip())

                inp_files = []
                for _ in range(2): 
                    inp_line = file_input.readline().strip()
                    inp_files.append(inp_line)

        except FileNotFoundError:
            print(f"Error: The file '{input_file_name}' was not found.")

        xp.random.seed(41)
        steps=Nvalues*nsample
        dim = 3
        tau=0
        kB =8.31036e-3
        dt=dwell/nsample  #The dwell time and sampling define the timestep
        v0=a/dt
        domain_size = [sn1, sn2, sn3]  #Lattice size

        dtime=sp.Symbol("dtime")
        kBoltz=sp.Symbol("kBoltz")
        Temp=sp.Symbol("Temp")


        #----------create data_handlings, arrays and fields and configure kernel env-------------

        if user_choice == 'gpu':            #data_handling that run on GPU
            dh = ps.create_data_handling(domain_size=domain_size, periodicity=periodicity, default_ghost_layers=1,default_target=target)
            x_arrays = dh.gpu_arrays
            config = ps.CreateKernelConfig(target=dh.default_target)  # if GPU, we configure kernel env that will run on GPU
            dcpu = ps.create_data_handling(domain_size=domain_size, periodicity=periodicity, default_ghost_layers=1)
            c_cpu = dcpu.add_array("c_cpu", values_per_cell=1)
            dcpu.fill(c_cpu.name, 0.0, ghost_layers=True)
            wij_cpu = dcpu.add_array("wij_cpu", values_per_cell=1)
            dcpu.fill(wij_cpu.name, 0.0, ghost_layers=True)

        if user_choice == 'cpu':         #data_handling that run on CPU
            dh = ps.create_data_handling(domain_size=domain_size, periodicity=periodicity, default_ghost_layers=1)
            x_arrays = dh.cpu_arrays
            config = ps.CreateKernelConfig(cpu_openmp=True)           # if CPU, we configure kernel env that will run on CPUs using OpenMP

        fields_to_create = {
            "c": 1, "c_next": 1,
            "e": 1,
            "latvacf": 3, "latvacf_next": 3,
            "latvacf_op": 3, "latvacf_next_op": 3,
            "Vmoy": 3, "Vmoy_next": 3,
            "probv0": 3, "probv0_next": 3,
            "locpfunc": 1, "locpfunc_next": 1,
            "Lattice": 1,
            "wij": 1,
            "ReG": 1, "ReG_next": 1,
            "ImG": 1, "ImG_next": 1,
            "ReGs": 1, "ReGs_next": 1,
            "ImGs": 1, "ImGs_next": 1,
        }

        for field_name, values_per_cell in fields_to_create.items():  #fill fields with 0.0 
            if user_choice == 'cpu':
                globals()[field_name] = dh.add_array(field_name, values_per_cell=values_per_cell, dtype=xp.float32, cpu=cpu, gpu=gpu)
                dh.fill(globals()[field_name].name, 0.0, ghost_layers=True)
            if user_choice == 'gpu':
                globals()[field_name] = dh.add_array(field_name, values_per_cell=values_per_cell)


        #------- creats arrays------------

        ReGtot = xp.zeros((nblocks + 1, steps), dtype=xp.float32)
        ImGtot = xp.zeros((nblocks + 1, steps), dtype=xp.float32)
        ReGtotsmp = xp.zeros((nblocks + 1, Nvalues), dtype=xp.float32)
        ImGtotsmp = xp.zeros((nblocks + 1, Nvalues), dtype=xp.float32)
        magnitude = xp.zeros((nblocks + 1, Nvalues), dtype=xp.float32)
        phase = xp.zeros((nblocks + 1, Nvalues), dtype=xp.float32)
        sumvacf = xp.zeros((nblocks + 1, steps), dtype=xp.float32)
        sumvacfx = xp.zeros((nblocks + 1, steps), dtype=xp.float32)
        sumvacfy = xp.zeros((nblocks + 1, steps), dtype=xp.float32)
        sumvacfz = xp.zeros((nblocks + 1, steps), dtype=xp.float32)
        denstot = xp.zeros((nblocks + 1, steps), dtype=xp.float32)
        lattvacfx = xp.zeros((nblocks + 1, steps), dtype=xp.float32)
        lattvacfy = xp.zeros((nblocks + 1, steps), dtype=xp.float32)
        lattvacfz = xp.zeros((nblocks + 1, steps), dtype=xp.float32)
        lattvacf = xp.zeros((nblocks + 1, steps), dtype=xp.float32)

        max_x_diff = 0
        max_y_diff = 0
        max_z_diff = 0

        for nb in range(1, nblocks + 1):
            x_diff = xmax[nb] - xmin[nb] - 4
            y_diff = ymax[nb] - ymin[nb] - 4
            z_diff = zmax[nb] - zmin[nb] - 4 
            xmax[nb]=int(xmax[nb])
            xmin[nb]=int(xmin[nb])
            ymax[nb]=int(ymax[nb])
            ymin[nb]=int(ymin[nb])
            zmax[nb]=int(zmax[nb])
            zmin[nb]=int(zmin[nb])

            if x_diff > max_x_diff:
                max_x_diff = x_diff
            if y_diff > max_y_diff:
                max_y_diff = y_diff
            if z_diff > max_z_diff:
                max_z_diff = z_diff

        xsumvacf = xp.zeros((int(nblocks + 1), int(max_x_diff + 1)), dtype=xp.float32)
        xsumvacfx = xp.zeros((int(nblocks + 1), int(max_x_diff + 1)), dtype=xp.float32)
        xsumvacfy = xp.zeros((int(nblocks + 1), int(max_x_diff + 1)), dtype=xp.float32)
        xsumvacfz = xp.zeros((int(nblocks + 1), int(max_x_diff + 1)), dtype=xp.float32)

        ysumvacf = xp.zeros((int(nblocks + 1), int(max_y_diff + 1)), dtype=xp.float32)
        ysumvacfx = xp.zeros((int(nblocks + 1), int(max_y_diff + 1)), dtype=xp.float32)
        ysumvacfy = xp.zeros((int(nblocks + 1), int(max_y_diff + 1)), dtype=xp.float32)
        ysumvacfz = xp.zeros((int(nblocks + 1), int(max_y_diff + 1)), dtype=xp.float32)

        zsumvacf = xp.zeros((int(nblocks + 1), int(max_z_diff + 1)), dtype=xp.float32)
        zsumvacfx = xp.zeros((int(nblocks + 1), int(max_z_diff + 1)), dtype=xp.float32)
        zsumvacfy = xp.zeros((int(nblocks + 1), int(max_z_diff + 1)), dtype=xp.float32)
        zsumvacfz = xp.zeros((int(nblocks + 1), int(max_z_diff + 1)), dtype=xp.float32)

        xvacf = xp.zeros((int(nblocks + 1), int(max_x_diff + 1)), dtype=xp.float32)
        xvacfx = xp.zeros((int(nblocks + 1), int(max_x_diff + 1)), dtype=xp.float32)
        xvacfy = xp.zeros((int(nblocks + 1), int(max_x_diff + 1)), dtype=xp.float32)
        xvacfz = xp.zeros((int(nblocks + 1), int(max_x_diff + 1)), dtype=xp.float32)

        yvacf = xp.zeros((int(nblocks + 1), int(max_y_diff + 1)), dtype=xp.float32)
        yvacfx = xp.zeros((int(nblocks + 1), int(max_y_diff + 1)), dtype=xp.float32)
        yvacfy = xp.zeros((int(nblocks + 1), int(max_y_diff + 1)), dtype=xp.float32)
        yvacfz = xp.zeros((int(nblocks + 1), int(max_y_diff + 1)), dtype=xp.float32)

        zvacf = xp.zeros((int(nblocks + 1), int(max_z_diff + 1)), dtype=xp.float32)
        zvacfx = xp.zeros((int(nblocks + 1), int(max_z_diff + 1)), dtype=xp.float32)
        zvacfy = xp.zeros((int(nblocks + 1), int(max_z_diff + 1)), dtype=xp.float32)
        zvacfz = xp.zeros((int(nblocks + 1), int(max_z_diff + 1)), dtype=xp.float32)



        #----------------------functions: ptr, fourrier transform, ....----------------
            
        def get_neighbors(dim,dirr):
            if (dim == 3 and dirr=='xyz') :
                return ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
            elif (dim == 3 and dirr=='x') :
                return ((1,0,0),(-1,0,0))
            elif (dim == 3 and dirr=='y') :
                return ((0,1,0),(0,-1,0))
            elif (dim == 3 and dirr=='z') :
                return ((0,0,1),(0,0,-1))
            else: raise ValueError(dim)

        def ptr(x,y,z):
            return sp.exp(-(x*sp.Abs(y)/z)/(sp.Symbol("kBoltz")*sp.Symbol("Temp")))

        def set_exp(x,y,z,w):
            return sp.Piecewise(((1-(sp.exp((-x)/(sp.Symbol("kBoltz")*sp.Symbol("Temp")))*ptr(y,z,w))), x >= 0),((1-ptr(y,z,w)), True))

        def modif_exp(x): 
            return sp.Min(1, sp.exp(x/(sp.Symbol("kBoltz")*sp.Symbol("Temp"))))

        def select_exp(x):
            return sp.Piecewise((sp.exp(x/(sp.Symbol("kBoltz")*sp.Symbol("Temp"))), x < 0), (1, True))

        def Vselect_exp(x):
            return sp.Piecewise((sp.exp(x), x < 0), (1, True))
        
        def locpfunc_exp(x,a,b,y,z,w):
            return sp.Piecewise(((b*ptr(y,z,w))+(a*(1-sp.exp((-x)/(sp.Symbol("kBoltz")*sp.Symbol("Temp"))))), x <= 0),((b*sp.exp(x/(sp.Symbol("kBoltz")*sp.Symbol("Temp")))+(a*(1-ptr(y,z,w)))), True))

        def fourier_transform(ReGtotsmp, ImGtotsmp, Nvalues, dwell, larmorfreq, tau, nsample, nb):
            global int_tau0, int_tau

            if user_choice == 'gpu':
                ReGtotsmp=cp.asnumpy(ReGtotsmp)
                ImGtotsmp=cp.asnumpy(ImGtotsmp)
                
            NMR_spec = np.zeros(Nvalues)
            NMR_freq = np.zeros(Nvalues)

            print(f"Doing FT.")

            name = f"FT_signal-{nsample}-bloc-{nb}.dat"
            with open(name, "w") as outFT:
                outFT.write("# Frequency - Magnitude - Phase - Real part - Imaginary part\n")

                count = 0
                for k in range(Nvalues // 2 + 1, Nvalues):
                    cos_values = np.cos(-2.0 * math.pi * k * np.arange(Nvalues) / Nvalues)
                    sin_values = np.sin(-2.0 * math.pi * k * np.arange(Nvalues) / Nvalues)
                    ReFT = np.dot(ReGtotsmp, cos_values) - np.dot(ImGtotsmp, sin_values)
                    ImFT = np.dot(ImGtotsmp, cos_values) + np.dot(ReGtotsmp, sin_values)

                    freq = -(Nvalues - k) / (Nvalues * dwell)
                    magnitude = np.sqrt(ReFT**2 + ImFT**2)
                    phase = np.arctan2(ImFT, ReFT)
                    outFT.write(f"{freq} {magnitude / Nvalues} {phase} {ReFT} {ImFT}\n")

                    NMR_spec[count] = magnitude
                    NMR_freq[count] = freq
                    count += 1

                for k in range(Nvalues // 2 + 1):
                    cos_values = np.cos(-2.0 * math.pi * k * np.arange(Nvalues) / Nvalues)
                    sin_values = np.sin(-2.0 * math.pi * k * np.arange(Nvalues) / Nvalues)
                    ReFT = np.dot(ReGtotsmp, cos_values) - np.dot(ImGtotsmp, sin_values)
                    ImFT = np.dot(ImGtotsmp, cos_values) + np.dot(ReGtotsmp, sin_values)

                    freq = k / (Nvalues * dwell)
                    magnitude = np.sqrt(ReFT**2 + ImFT**2)
                    phase = np.arctan2(ImFT, ReFT)
                    outFT.write(f"{freq} {magnitude / Nvalues} {phase} {ReFT} {ImFT}\n")

                    NMR_spec[count] = magnitude
                    NMR_freq[count] = freq
                    count += 1

            minvalue = np.min(NMR_spec)
            NMR_spec -= minvalue

            save = np.trapz(NMR_spec, NMR_freq)

            name = f"Normalised_spectrum-{nsample}-bloc-{nb}.dat"
            with open(name, "w") as outFT:
                outFT.write("# Frequency (ppm) - Magnitude\n")
                NMR_spec *= larmorfreq / save
                NMR_freq /= larmorfreq
                maxvalue = np.max(NMR_spec)
                np.savetxt(outFT, np.column_stack((NMR_freq, NMR_spec)), fmt='%f %f')

            k = np.argmax(NMR_spec >= maxvalue / 2.0)
            firstfreq = (NMR_freq[k] + NMR_freq[k - 1]) / 2.0

            k = np.argmax(NMR_spec[::-1] >= maxvalue / 2.0)
            secondfreq = (NMR_freq[Nvalues - 1 - k] + NMR_freq[Nvalues - k]) / 2.0

            halfwidth_ppm = secondfreq - firstfreq
            halfwidth_hz = halfwidth_ppm * larmorfreq

            if tau == 0:
                int_tau0 = save
                print(f"FT done-bloc-{nb} (initial integral = {int_tau0}).")
            elif tau > 0:
                int_tau = save
                print(f"FT done-bloc-{nb} (initial integral = {int_tau}).")

            print(f"Halfwidth (ppm)-bloc-{nb}: {halfwidth_ppm}.")
            print(f"Halfwidth (Hz)-bloc-{nb}: {halfwidth_hz}.")
            del NMR_spec, NMR_freq, ReGtotsmp, ImGtotsmp
            
            return

        #-----------------------------------------Non periodic conditions and layers synchronization------------------------------------------------
        def periodic_condition(field_1,field_2,field_3,field_4):
            if not periodicity[0]:
                field_1[[0, -1, 1, -2],1:sn2+1, 1:sn3+1] = 1
                field_2[[0, -1, 1, -2],1:sn2+1, 1:sn3+1] = 0.0
                field_3[[0, -1, 1, -2],1:sn2+1, 1:sn3+1] = 0.0
                field_4[[0, -1, 1, -2],1:sn2+1, 1:sn3+1] = 0.0
            if not periodicity[1]:
                field_1[1:sn1+1, [0, -1, 1, -2], 1:sn3+1] = 1
                field_2[1:sn1+1, [0, -1, 1, -2], 1:sn3+1] = 0.0
                field_3[1:sn1+1, [0, -1, 1, -2], 1:sn3+1] = 0.0
                field_4[1:sn1+1, [0, -1, 1, -2], 1:sn3+1] = 0.0
            if not periodicity[2]:
                field_1[1:sn1+1, 1:sn2+1, [0, -1, 1, -2]] = 1
                field_2[1:sn1+1, 1:sn2+1, [0, -1, 1, -2]] = 0.0
                field_3[1:sn1+1, 1:sn2+1, [0, -1, 1, -2]] = 0.0
                field_4[1:sn1+1, 1:sn2+1, [0, -1, 1, -2]] = 0.0
            return field_1, field_2, field_3, field_4

        sync1 = dh.synchronization_function([c.name])       #create sync functions for fields for periodic axis
        sync2 = dh.synchronization_function([e.name])
        sync3 = dh.synchronization_function([probv0.name])
        sync4 = dh.synchronization_function([ReG.name])
        sync5 = dh.synchronization_function([ImG.name])
        sync6 = dh.synchronization_function([Lattice.name])

        #------------------Numba_CPU_parallel function to fill 3D arrays: densities, Energies, frequencies------------------

        @njit(nopython=True, parallel=True, fastmath=True)
        def process_data(xmin, xmax, ymin, ymax, zmin, zmax, c, w, pore_size_data, density, shift):
            for i in prange(xmin, xmax+1):
                for j in range(ymin, ymax+1):
                    for k in range(zmin, zmax+1):
                        pore_val = c[i, j, k]

                        differences = np.abs(pore_size_data - pore_val)

                        idx = np.argmin(differences)

                        c[i, j, k] = density[idx]
                        w[i, j, k] = shift[idx] * larmorfreq * 2.0 * math.pi
                

        num_threads_str = os.getenv('OMP_NUM_THREADS', '1')
        num_threads = int(num_threads_str)
        set_num_threads(num_threads)
    comm.Barrier()

    if rank ==0:
        pore_size_list = np.unique(np.loadtxt(pore_prob_file, usecols=0, comments='#', ndmin=1)).tolist()
    comm.Barrier()
    
    if rank == 0:
        user_choice = user_choice
    else:
        user_choice = None

    pore_size_list = comm.bcast(pore_size_list, root=0)
    zmat_files = comm.bcast(zmat_files, root=0)
    zmat_values = comm.bcast(zmat_values, root=0)
    inp_files = comm.bcast(inp_files, root=0)
    user_choice = comm.bcast(user_choice, root=0)
    use_gpu = (user_choice == "gpu")         
    ngpu = int(os.getenv("LPC3D_NUM_GPUS", "1"))
    
    all_dens = mpi4py_lammps.run_simulation_lammps(pore_size_list, zmat_files, zmat_values, inp_files, comm, rank, size, use_gpu, ngpu)
    arr_dens = np.asarray(all_dens, dtype=float)

    if rank==0:
        with open("all_densities.dat", "w") as f:
            f.write("# pore_size  M1_s1  M1_s2  M2_s1  M2_s2  M3_s1  M3_s2\n")
            for i, psize in enumerate(pore_size_list):
                # M1 = 0, M2 = 1, M3 = 2
                M1_s1 = arr_dens[i, 0, 1]  # slice 1
                M1_s2 = arr_dens[i, 0, 3]  # slice 2

                M2_s1 = arr_dens[i, 1, 1]
                M2_s2 = arr_dens[i, 1, 3]

                M3_s1 = arr_dens[i, 2, 1]
                M3_s2 = arr_dens[i, 2, 3]

                f.write(f"{psize:12.6g} {M1_s1:15.8e} {M1_s2:15.8e} "
                        f"{M2_s1:15.8e} {M2_s2:15.8e} "
                        f"{M3_s1:15.8e} {M3_s2:15.8e}\n")
    comm.Barrier()
        #--------------------- fill quantities in arrays and fields------------------------------------	
    
    if rank==0:

        for nb in range(1,nblocks+1):
            if block_names[nb]=='electrode':
                data_pore = np.loadtxt(pore_prob_file)
                data_shift = np.loadtxt(pore_shift_file[nb])

                size_pores = data_pore[:, 0]
                probability = data_pore[:, 1]
                probability /= probability.sum()

                pore_size_data = data_shift[:, 0]
        

                dens_slice  = arr_dens[:, calculated_spec-1, nb]    
                shift_slice = data_shift[:, calculated_spec]        

                k1=int(xmax[nb] - xmin[nb] + 1)
                k2=int(ymax[nb] - ymin[nb] + 1)
                k3=int(zmax[nb] - zmin[nb] + 1)

                if user_choice == 'cpu':   

                    x_arrays[c.name][xmin[nb]:xmax[nb]+1,ymin[nb]:ymax[nb]+1,zmin[nb]:zmax[nb]+1] = np.random.choice(size_pores, size=(k1,k2,k3), p=probability)
                    process_data(xmin[nb], xmax[nb], ymin[nb], ymax[nb], zmin[nb], zmax[nb], x_arrays[c.name], x_arrays[wij.name],pore_size_data, dens_slice, shift_slice)

                

                if user_choice == 'gpu':
                    dcpu.cpu_arrays[c_cpu.name][xmin[nb]:xmax[nb]+1,ymin[nb]:ymax[nb]+1,zmin[nb]:zmax[nb]+1] = np.random.choice(size_pores, size=(k1,k2,k3), p=probability)
                    process_data(xmin[nb], xmax[nb], ymin[nb], ymax[nb], zmin[nb], zmax[nb], dcpu.cpu_arrays[c_cpu.name], dcpu.cpu_arrays[wij_cpu.name],pore_size_data, dens_slice, shift_slice)
                    x_arrays[c.name]=cp.array(dcpu.cpu_arrays[c_cpu.name])
                    x_arrays[wij.name]=cp.array(dcpu.cpu_arrays[wij_cpu.name])

                
            if block_names[nb]=='bulk':
                x_arrays[wij.name][xmin[nb]:xmax[nb]+1,ymin[nb]:ymax[nb]+1,zmin[nb]:zmax[nb]+1] = freq_bulk[nb]* larmorfreq * 2.0 * math.pi
                x_arrays[c.name][xmin[nb]:xmax[nb]+1,ymin[nb]:ymax[nb]+1,zmin[nb]:zmax[nb]+1] = dens_bulk[nb]
                bulk=dens_bulk[nb]
    
    comm.Barrier()
 
   
    
    if rank==0:
        if electrode_nature=='particles':
            with open(particles_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    atom_type = parts[0]  # 'B' or'P'
                    x, y, z = int(parts[1]), int(parts[2]), int(parts[3])  

                    if atom_type == 'B':
                        x_arrays[c.name][x, y, z] = bulk
                        x_arrays[wij.name][x, y, z] = 0.0 
                                    

        for nb in range(1,nblocks+1):
            if block_names[nb]=='electrode':
                x_arrays[e.name][xmin[nb]:xmax[nb]+1,ymin[nb]:ymax[nb]+1,zmin[nb]:zmax[nb]+1] = -kB*T*xp.log(x_arrays[c.name][xmin[nb]:xmax[nb]+1,ymin[nb]:ymax[nb]+1,zmin[nb]:zmax[nb]+1]/xp.sum(x_arrays[c.name][1:sn1+1,1:sn2+1,1:sn3+1]))
            if block_names[nb]=='electrode': 
                if elec_fil[nb]==0:
                    x_arrays[c.name][xmin[nb]:xmax[nb]+1,ymin[nb]:ymax[nb]+1,zmin[nb]:zmax[nb]+1]=0.0  
            if block_names[nb]=='bulk':
                x_arrays[e.name][xmin[nb]:xmax[nb]+1,ymin[nb]:ymax[nb]+1,zmin[nb]:zmax[nb]+1] = -kB*T*xp.log(dens_bulk[nb]/xp.sum(x_arrays[c.name][1:sn1+1,1:sn2+1,1:sn3+1]))

        periodic_condition(x_arrays[Lattice.name],x_arrays[c.name],x_arrays[e.name],x_arrays[wij.name])

        #----------------equilibration process-------------

        center = tuple([0]*dim)
        initial = c.center
        direct=sp.Rational(1.0,len(get_neighbors(dim,'xyz')))
        sync2()                                                        # synchronize ghost layer in periodic conditions
        sync6()

        if equilibration==1:

            what_arrives  = sum([(1-Lattice[center])*direct*c[n] *modif_exp(-(e[center]-e[n]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])
            what_leaves   = sum([(1-Lattice[n])*(1-Lattice[center])*direct*c[center] *modif_exp(-(e[n]-e[center]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])

            ur =ps.AssignmentCollection(subexpressions=[ps.Assignment(kBoltz, kB),ps.Assignment(Temp, T)],   #create update rule
                main_assignments= [ps.Assignment(c_next.center, initial + what_arrives - what_leaves)])
            ur=ps.simp.insert_constants(ur)   
            ast = ps.create_kernel(ur, config=config)                      # create the kernel for the update rule with the configuration (cpu or gpu)
            kernel = ast.compile()                                         #compile the kernel
            verif=1
            vtk_writer = dh.create_vtk_writer('Density_Energy_out', ['c', 'e']) #create vtk function for vizualisation

        
        #------------------ main loop for equilibration process: calculate densities in each time step--------------

            vtk_writer(1)        
            time_loop(nsteps_equ) 
            vtk_writer(2)
            if verif==1:
                print("System still not equilibrated! increase the step number")
                sys.exit()

        #----------------propagation--------------
        #----------------calculation at t=0-------
        #---------------latvacf in 3 directions calculation-------------


        sync1()
        # create the assignement for latvacf using symbols with 'SymPy'
        lattvacfx_calc  = sum([(1-Lattice[center])*(1-Lattice[n])*direct*v0*v0*c[center]*select_exp(-(e[n]-e[center]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'x')])
        lattvacfy_calc  = sum([(1-Lattice[center])*(1-Lattice[n])*direct*v0*v0*c[center]*select_exp(-(e[n]-e[center]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'y')])
        lattvacfz_calc  = sum([(1-Lattice[center])*(1-Lattice[n])*direct*v0*v0*c[center]*select_exp(-(e[n]-e[center]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'z')])

        ur =ps.AssignmentCollection(subexpressions=[ps.Assignment(kBoltz, kB),ps.Assignment(Temp, T)], 
            main_assignments= [ps.Assignment(latvacf_next[0,0,0][0], latvacf[0,0,0][0] + lattvacfx_calc),
            ps.Assignment(latvacf_next[0,0,0][1], latvacf[0,0,0][1] + lattvacfy_calc),
            ps.Assignment(latvacf_next[0,0,0][2], latvacf[0,0,0][2] + lattvacfz_calc)])

        ur=ps.simp.insert_constants(ur)
        ast = ps.create_kernel(ur, config=config)
        kernel = ast.compile()

        dh.run_kernel(kernel)
        dh.swap(latvacf_next.name, latvacf.name)


        # we calculate vacf in all directions (6 directions)
        for nb in range(1, nblocks+1):
            xvacfx[nb] = xp.sum(x_arrays[latvacf.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0], axis=(1, 2))
            yvacfx[nb] = xp.sum(x_arrays[latvacf.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0], axis=(0, 2))
            zvacfx[nb] = xp.sum(x_arrays[latvacf.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0], axis=(0, 1))
            xvacfy[nb] = xp.sum(x_arrays[latvacf.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,1], axis=(1, 2))
            yvacfy[nb] = xp.sum(x_arrays[latvacf.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,1], axis=(0, 2))
            zvacfy[nb] = xp.sum(x_arrays[latvacf.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,1], axis=(0, 1))
            xvacfz[nb] = xp.sum(x_arrays[latvacf.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,2], axis=(1, 2))
            yvacfz[nb] = xp.sum(x_arrays[latvacf.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,2], axis=(0, 2))
            zvacfz[nb] = xp.sum(x_arrays[latvacf.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,2], axis=(0, 1))

            xvacf[nb]= xvacfx[nb] + xvacfy[nb] + xvacfz[nb]
            yvacf[nb] = yvacfx[nb] + yvacfy[nb] + yvacfz[nb]
            zvacf[nb] = zvacfx[nb] + zvacfy[nb] + zvacfz[nb]

            xsumvacf[nb] += xvacf[nb] * dt / 2.0; xsumvacfx[nb] += xvacfx[nb] * dt / 2.0; xsumvacfy[nb] += xvacfy[nb] * dt / 2.0; xsumvacfz[nb] += xvacfz[nb] * dt / 2.0
            ysumvacf[nb] += yvacf[nb] * dt / 2.0; ysumvacfx[nb] += yvacfx[nb] * dt / 2.0; ysumvacfy[nb] += yvacfy[nb] * dt / 2.0; ysumvacfz[nb] += yvacfz[nb] * dt / 2.0
            zsumvacf[nb] += zvacf[nb] * dt / 2.0; zsumvacfx[nb] += zvacfx[nb] * dt / 2.0; zsumvacfy[nb] += zvacfy[nb] * dt / 2.0; zsumvacfz[nb] += zvacfz[nb] * dt / 2.0

            sumvacf[nb, 0]=0; sumvacfx[nb, 0]=0; sumvacfy[nb, 0]=0; sumvacfz[nb, 0]=0
            denstot[nb, 0] = xp.sum(x_arrays[c.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2])
            lattvacfx[nb, 0]+= xp.sum(x_arrays[latvacf.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0])
            lattvacfy[nb, 0]+= xp.sum(x_arrays[latvacf.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,1])
            lattvacfz[nb, 0]+= xp.sum(x_arrays[latvacf.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,2])
            lattvacf[nb, 0]+= lattvacfx[nb, 0] + lattvacfy[nb, 0] + lattvacfz[nb, 0]
            sumvacf[nb, 0]+= (lattvacf[nb, 0]*dt/(2.0*denstot[nb, 0]))
            sumvacfx[nb, 0]+= (lattvacfx[nb, 0]*dt/(2.0*denstot[nb, 0]))
            sumvacfy[nb, 0]+= (lattvacfy[nb, 0]*dt/(2.0*denstot[nb, 0]))
            sumvacfz[nb, 0]+= (lattvacfz[nb, 0]*dt/(2.0*denstot[nb, 0]))

            ReGtot[nb, 0]=1.0; ImGtot[nb, 0]=0.0; ReGtotsmp[nb, 0]=1.0; ImGtotsmp[nb, 0]=0.0
            magnitude[nb, 0] = xp.sqrt(ReGtot[nb, 0]**2 + ImGtot[nb, 0]**2)
            phase[nb, 0] = xp.arctan2(ImGtot[nb, 0], ReGtot[nb, 0])

        #7----------------calculation at t=1-------
        #7-1---------------vmoyen calculation---------------------
        #7-2---------------probv0 and locpfun  and density calculation---------------------

        vxmoy_calc  = (1-Lattice[center])*(1-Lattice[1,0,0])*direct*v0*Vselect_exp(-(e[1,0,0]-e[center]))*ptr(Ea,(c[center]-c[1,0,0]),(c[center]+c[1,0,0])) - (1-Lattice[center])*(1-Lattice[-1,0,0])*direct*v0*Vselect_exp(-(e[-1,0,0]-e[center]))*ptr(Ea,(c[center]-c[-1,0,0]),(c[center]+c[-1,0,0]))
        vymoy_calc  = (1-Lattice[center])*(1-Lattice[0,1,0])*direct*v0*Vselect_exp(-(e[0,1,0]-e[center]))*ptr(Ea,(c[center]-c[0,1,0]),(c[center]+c[0,1,0])) - (1-Lattice[center])*(1-Lattice[0,-1,0])*direct*v0*Vselect_exp(-(e[0,-1,0]-e[center]))*ptr(Ea,(c[center]-c[0,-1,0]),(c[center]+c[0,-1,0]))
        vzmoy_calc  = (1-Lattice[center])*(1-Lattice[0,0,1])*direct*v0*Vselect_exp(-(e[0,0,1]-e[center]))*ptr(Ea,(c[center]-c[0,0,1]),(c[center]+c[0,0,1])) - (1-Lattice[center])*(1-Lattice[0,0,-1])*direct*v0*Vselect_exp(-(e[0,0,-1]-e[center]))*ptr(Ea,(c[center]-c[0,0,-1]),(c[center]+c[0,0,-1]))
        probv0x_calc  = (1-Lattice[center])*(1-Lattice[-1,0,0])*c[-1,0,0]*v0*select_exp(-(e[center]-e[-1,0,0]))*ptr(Ea,(c[center]-c[-1,0,0]),(c[center]+c[-1,0,0])) - (1-Lattice[center])*(1-Lattice[1,0,0])*c[1,0,0]*v0*select_exp(-(e[center]-e[1,0,0]))*ptr(Ea,(c[center]-c[1,0,0]),(c[center]+c[1,0,0]))
        probv0y_calc  = (1-Lattice[center])*(1-Lattice[0,-1,0])*c[0,-1,0]*v0*select_exp(-(e[center]-e[0,-1,0]))*ptr(Ea,(c[center]-c[0,-1,0]),(c[center]+c[0,-1,0])) - (1-Lattice[center])*(1-Lattice[0,1,0])*c[0,1,0]*v0*select_exp(-(e[center]-e[0,1,0]))*ptr(Ea,(c[center]-c[0,1,0]),(c[center]+c[0,1,0]))
        probv0z_calc  = (1-Lattice[center])*(1-Lattice[0,0,-1])*c[0,0,-1]*v0*select_exp(-(e[center]-e[0,0,-1]))*ptr(Ea,(c[center]-c[0,0,-1]),(c[center]+c[0,0,-1])) - (1-Lattice[center])*(1-Lattice[0,0,1])*c[0,0,1]*v0*select_exp(-(e[center]-e[0,0,1]))*ptr(Ea,(c[center]-c[0,0,1]),(c[center]+c[0,0,1]))
        locpfunc_calc  = sum([(1-Lattice[center])*(1-Lattice[n])*locpfunc_exp((-(e[center]-e[n])), c[center], c[n],Ea,(c[center]-c[n]),(c[center]+c[n])) + (c[center]*Lattice[n]) for n in get_neighbors(dim,'xyz')])

        what_arrives  = sum([(1-Lattice[center])*direct*c[n] *modif_exp(-(e[center]-e[n]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])
        what_leaves   = sum([(1-Lattice[center])*(1-Lattice[n])*direct*c[center] *modif_exp(-(e[n]-e[center]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])

        ur =ps.AssignmentCollection(subexpressions=[ps.Assignment(kBoltz, kB),ps.Assignment(Temp, T)], 
            main_assignments= [ps.Assignment(Vmoy_next[0,0,0][0], Vmoy[0,0,0][0] + vxmoy_calc),
            ps.Assignment(Vmoy_next[0,0,0][1], Vmoy[0,0,0][1] + vymoy_calc),
            ps.Assignment(Vmoy_next[0,0,0][2], Vmoy[0,0,0][2] + vzmoy_calc),
            ps.Assignment(probv0_next[0,0,0][0], probv0[0,0,0][0] + probv0x_calc),
            ps.Assignment(probv0_next[0,0,0][1], probv0[0,0,0][1] + probv0y_calc),
            ps.Assignment(probv0_next[0,0,0][2], probv0[0,0,0][2] + probv0z_calc),
            ps.Assignment(locpfunc_next.center, locpfunc.center + locpfunc_calc),
            ps.Assignment(c_next.center, initial + what_arrives - what_leaves)])

        ur=ps.simp.insert_constants(ur)
        ast = ps.create_kernel(ur, config=config)
        kernel = ast.compile()

        dh.run_kernel(kernel)

        dh.swap(Vmoy.name, Vmoy_next.name)
        dh.swap(probv0.name, probv0_next.name)
        dh.swap(locpfunc.name, locpfunc_next.name)
        dh.swap(c.name, c_next.name)

        ur1 =ps.AssignmentCollection(subexpressions=[ps.Assignment(dtime, dt)], 
            main_assignments= [ps.Assignment(probv0.center[0],probv0.center[0]/ locpfunc.center),
            ps.Assignment(probv0.center[1], probv0.center[1]/ locpfunc.center),
            ps.Assignment(probv0.center[2], probv0.center[2]/ locpfunc.center),
            ps.Assignment(ReG.center, (1-Lattice[center])*sp.cos(wij.center*sp.Symbol("dtime"))),
            ps.Assignment(ImG.center, (1-Lattice[center])*sp.sin(wij.center*sp.Symbol("dtime")))])

        ur2= [ps.Assignment(ReGs.center, (1-Lattice[center])*c.center*ReG.center),
            ps.Assignment(ImGs.center, (1-Lattice[center])*c.center*ImG.center),
            ps.Assignment(latvacf_op.center[0], c.center*probv0[0,0,0][0]*Vmoy[0,0,0][0]),
            ps.Assignment(latvacf_op.center[1], c.center*probv0[0,0,0][1]*Vmoy[0,0,0][1]),
            ps.Assignment(latvacf_op.center[2], c.center*probv0[0,0,0][2]*Vmoy[0,0,0][2])]

        ur1=ps.simp.insert_constants(ur1)
        ast1 = ps.create_kernel(ur1, config=config)
        kernel1 = ast1.compile()
        dh.run_kernel(kernel1)
        x_arrays[probv0.name]=xp.nan_to_num(x_arrays[probv0.name], nan=0.0)

        ast2 = ps.create_kernel(ur2, config=config)
        kernel2 = ast2.compile()
        dh.run_kernel(kernel2)


        for nb in range(1, nblocks+1):
            lattvacfx[nb, 1]= xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0])
            lattvacfy[nb, 1]= xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,1])
            lattvacfz[nb, 1]= xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,2])

            lattvacf[nb, 1] = lattvacfx[nb, 1]+lattvacfy[nb, 1]+lattvacfz[nb, 1]

            xvacfx[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0], axis=(1,2))
            xvacfy[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,1], axis=(1,2))
            xvacfz[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,2], axis=(1,2))

            xvacf[nb]= xvacfx[nb] + xvacfy[nb] + xvacfz[nb]

            yvacfx[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0], axis=(0,2))
            yvacfy[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,1], axis=(0,2))
            yvacfz[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,2], axis=(0,2))

            yvacf[nb] = yvacfx[nb] + yvacfy[nb] + yvacfz[nb]

            zvacfx[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0], axis=(0,1))
            zvacfy[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0], axis=(0,1))
            zvacfz[nb]=xp.sum(x_arrays[latvacf_op.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2,0], axis=(0,1))

            zvacf[nb] = zvacfx[nb] + zvacfy[nb] + zvacfz[nb]

            denstot[nb, 1]=xp.sum(x_arrays[c.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2])
            ReGtot[nb, 1]=xp.sum(x_arrays[ReGs.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2])/denstot[nb, 1]
            ImGtot[nb, 1]=xp.sum(x_arrays[ImGs.name][xmin[nb]+2:xmax[nb]+1-2,ymin[nb]+2:ymax[nb]+1-2,zmin[nb]+2:zmax[nb]+1-2])/denstot[nb, 1]

            xsumvacf[nb] += xvacf[nb] * dt / 2.0; xsumvacfx[nb] += xvacfx[nb] * dt / 2.0; xsumvacfy[nb] += xvacfy[nb] * dt / 2.0; xsumvacfz[nb] += xvacfz[nb] * dt / 2.0
            ysumvacf[nb] += yvacf[nb] * dt / 2.0; ysumvacfx[nb] += yvacfx[nb] * dt / 2.0; ysumvacfy[nb] += yvacfy[nb] * dt / 2.0; ysumvacfz[nb] += yvacfz[nb] * dt / 2.0
            zsumvacf[nb] += zvacf[nb] * dt / 2.0; zsumvacfx[nb] += zvacfx[nb] * dt / 2.0; zsumvacfy[nb] += zvacfy[nb] * dt / 2.0; zsumvacfz[nb] += zvacfz[nb] * dt / 2.0

            if nsample==1:
                magnitude[nb, 1] = xp.sqrt(ReGtot[nb, 1]**2 + ImGtot[nb, 1]**2)
                phase[nb, 1] = xp.arctan2(ImGtot[nb, 1], ReGtot[nb, 1])
                ReGtotsmp[nb, 1]=ReGtot[nb, 1]
                ImGtotsmp[nb, 1]=ImGtot[nb, 1]
                ireal=2
            if nsample > 1:
                ireal=1
            sumvacf[nb, 1]= sumvacf[nb, 0] + ((lattvacf[nb, 1]*dt)/(denstot[nb, 1]))
            sumvacfx[nb, 1]= sumvacfx[nb, 0] + ((lattvacfx[nb, 1]*dt)/(denstot[nb, 1]))
            sumvacfy[nb, 1]= sumvacfy[nb, 0] + ((lattvacfy[nb, 1]*dt)/(denstot[nb, 1]))
            sumvacfz[nb, 1]= sumvacfz[nb, 0] + ((lattvacfz[nb, 1]*dt)/(denstot[nb, 1]))

        #---------------calculation at t>1--------
        #---------------the main loop for propagation process-------------
        #---------------calculate all quantities----------------

        #--create expressions for the quantities
        what_arrives_probv0x= sum([(1-Lattice[center])*(1-Lattice[n])*c[n]*probv0[n][0]*modif_exp(-(e[center]-e[n]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])  
        what_arrives_probv0y= sum([(1-Lattice[center])*(1-Lattice[n])*c[n]*probv0[n][1]*modif_exp(-(e[center]-e[n]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])
        what_arrives_probv0z= sum([(1-Lattice[center])*(1-Lattice[n])*c[n]*probv0[n][2]*modif_exp(-(e[center]-e[n]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])

        what_arrives_ReG= sum([(1-Lattice[center])*(1-Lattice[n])*(c[n]*ReG[n]*modif_exp(-(e[center]-e[n]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n]))) for n in get_neighbors(dim,'xyz')])
        what_arrives_ImG= sum([(1-Lattice[center])*(1-Lattice[n])*(c[n]*ImG[n]*modif_exp(-(e[center]-e[n]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n]))) for n in get_neighbors(dim,'xyz')])
        what_arrives_locpfunc= sum([(1-Lattice[center])*(1-Lattice[n])*c[n]*modif_exp(-(e[center]-e[n]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])

        what_stays_probv0x= sum([(1-Lattice[center])*(1-Lattice[n])*c[center]*probv0[center][0]*set_exp(-(e[center]-e[n]),Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])
        what_stays_probv0xx= sum([(1-Lattice[center])*c[center]*probv0[center][0]*Lattice[n] for n in get_neighbors(dim,'xyz')])
        what_stays_probv0y= sum([(1-Lattice[center])*(1-Lattice[n])*c[center]*probv0[center][1]*set_exp(-(e[center]-e[n]),Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])
        what_stays_probv0yy= sum([(1-Lattice[center])*c[center]*probv0[center][1]*Lattice[n] for n in get_neighbors(dim,'xyz')])
        what_stays_probv0z= sum([(1-Lattice[center])*(1-Lattice[n])*c[center]*probv0[center][2]*set_exp(-(e[center]-e[n]),Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])
        what_stays_probv0zz= sum([(1-Lattice[center])*c[center]*probv0[center][2]*Lattice[n] for n in get_neighbors(dim,'xyz')])

        what_stays_ReG= sum([(1-Lattice[center])*(1-Lattice[n])*(c[center]*ReG[center]*set_exp(-(e[center]-e[n]),Ea,(c[center]-c[n]),(c[center]+c[n]))) for n in get_neighbors(dim,'xyz')])
        what_stays_ImG= sum([(1-Lattice[center])*(1-Lattice[n])*(c[center]*ImG[center]*set_exp(-(e[center]-e[n]),Ea,(c[center]-c[n]),(c[center]+c[n]))) for n in get_neighbors(dim,'xyz')])

        what_stays_ReGxx= sum([(1-Lattice[center])*(c[center]*ReG[center]*Lattice[n]) for n in get_neighbors(dim,'xyz')])
        what_stays_ImGxx= sum([(1-Lattice[center])*(c[center]*ImG[center]*Lattice[n]) for n in get_neighbors(dim,'xyz')])

        what_stays_locpfunc= sum([(1-Lattice[center])*(1-Lattice[n])*c[center]*set_exp(-(e[center]-e[n]),Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])
        what_stays_locpfuncxx= sum([(1-Lattice[center])*c[center]*Lattice[n] for n in get_neighbors(dim,'xyz')])

        what_arrives_c  = sum([(1-Lattice[center])*(1-Lattice[n])*direct*c[n] *modif_exp(-(e[center]-e[n]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])
        what_leaves_c   = sum([(1-Lattice[center])*(1-Lattice[n])*direct*c[center] *modif_exp(-(e[n]-e[center]))*ptr(Ea,(c[center]-c[n]),(c[center]+c[n])) for n in get_neighbors(dim,'xyz')])

        vxmoy_calc  = (1-Lattice[center])*(1-Lattice[1,0,0])*direct*v0*Vselect_exp(-(e[1,0,0]-e[center]))*ptr(Ea,(c[center]-c[1,0,0]),(c[center]+c[1,0,0])) - (1-Lattice[center])*(1-Lattice[-1,0,0])*direct*v0*Vselect_exp(-(e[-1,0,0]-e[center]))*ptr(Ea,(c[center]-c[-1,0,0]),(c[center]+c[-1,0,0]))
        vymoy_calc  = (1-Lattice[center])*(1-Lattice[0,1,0])*direct*v0*Vselect_exp(-(e[0,1,0]-e[center]))*ptr(Ea,(c[center]-c[0,1,0]),(c[center]+c[0,1,0])) - (1-Lattice[center])*(1-Lattice[0,-1,0])*direct*v0*Vselect_exp(-(e[0,-1,0]-e[center]))*ptr(Ea,(c[center]-c[0,-1,0]),(c[center]+c[0,-1,0]))
        vzmoy_calc  = (1-Lattice[center])*(1-Lattice[0,0,1])*direct*v0*Vselect_exp(-(e[0,0,1]-e[center]))*ptr(Ea,(c[center]-c[0,0,1]),(c[center]+c[0,0,1])) - (1-Lattice[center])*(1-Lattice[0,0,-1])*direct*v0*Vselect_exp(-(e[0,0,-1]-e[center]))*ptr(Ea,(c[center]-c[0,0,-1]),(c[center]+c[0,0,-1]))

        #--create kernels for those expressions
        symd=ps.AssignmentCollection(subexpressions=[ps.Assignment(kBoltz, kB),ps.Assignment(Temp, T)], 
            main_assignments=[ps.Assignment(probv0_next.center[0],  what_arrives_probv0x + what_stays_probv0x + what_stays_probv0xx), 
            ps.Assignment(probv0_next.center[1],  what_arrives_probv0y + what_stays_probv0y + what_stays_probv0yy),
            ps.Assignment(probv0_next.center[2],  what_arrives_probv0z + what_stays_probv0z + what_stays_probv0zz),
            ps.Assignment(ReG_next.center,  what_arrives_ReG + what_stays_ReG + what_stays_ReGxx),
            ps.Assignment(ImG_next.center,  what_arrives_ImG + what_stays_ImG + what_stays_ImGxx),
            ps.Assignment(locpfunc_next.center, what_arrives_locpfunc + what_stays_locpfunc + what_stays_locpfuncxx),
            ps.Assignment(c_next.center, initial + what_arrives_c - what_leaves_c),
            ps.Assignment(Vmoy_next[0,0,0][0], vxmoy_calc),
            ps.Assignment(Vmoy_next[0,0,0][1], vymoy_calc),
            ps.Assignment(Vmoy_next[0,0,0][2], vzmoy_calc)])

        symd=ps.simp.insert_constants(symd)
        ast = ps.create_kernel(symd, config=config)
        kernel = ast.compile()

        ur1 = [ps.Assignment(ReG[0,0,0], (1-Lattice[center])*ReG[0,0,0] / locpfunc.center),
            ps.Assignment(ImG[0,0,0], (1-Lattice[center])*ImG[0,0,0] / locpfunc.center)]

        ur2=ps.AssignmentCollection(subexpressions=[ps.Assignment(dtime, dt)], 
            main_assignments=[ps.Assignment(ReGs.center, ((1-Lattice[center])*ReG.center*sp.cos(wij.center*sp.Symbol("dtime")))-((1-Lattice[center])*ImG.center*sp.sin(wij.center*sp.Symbol("dtime")))),
            ps.Assignment(ImGs.center, ((1-Lattice[center])*ReG.center*sp.sin(wij.center*sp.Symbol("dtime")))+((1-Lattice[center])*ImG.center*sp.cos(wij.center*sp.Symbol("dtime"))))])
        ur2=ps.simp.insert_constants(ur2)

        ur3=[ps.Assignment(ReG[0,0,0], ReGs[0,0,0]),
            ps.Assignment(ImG[0,0,0], ImGs[0,0,0]),
            ps.Assignment(probv0[0,0,0][0], (1-Lattice[center])*probv0[0,0,0][0] / locpfunc.center),
            ps.Assignment(probv0[0,0,0][1], (1-Lattice[center])*probv0[0,0,0][1] / locpfunc.center),
            ps.Assignment(probv0[0,0,0][2], (1-Lattice[center])*probv0[0,0,0][2] / locpfunc.center)]

        ur4= [ps.Assignment(latvacf_op.center[0], c.center*probv0[0,0,0][0]*Vmoy[0,0,0][0]),
            ps.Assignment(latvacf_op.center[1], c.center*probv0[0,0,0][1]*Vmoy[0,0,0][1]),
            ps.Assignment(latvacf_op.center[2], c.center*probv0[0,0,0][2]*Vmoy[0,0,0][2]),
            ps.Assignment(ReGs.center, c.center*ReG.center),
            ps.Assignment(ImGs.center, c.center*ImG.center)]

        #--compile kernels for those expressions
        ast1 = ps.create_kernel(ur1, config=config)
        kernel1 = ast1.compile()
        ast2 = ps.create_kernel(ur2, config=config)
        kernel2 = ast2.compile()
        ast3 = ps.create_kernel(ur3, config=config)
        kernel3 = ast3.compile()
        ast4 = ps.create_kernel(ur4, config=config)
        kernel4 = ast4.compile()

        #---------- propagation function-------
        
                
        print("propagation") 
        if steps <= 100:
            display_interval = 10
        elif steps < 5000:
            display_interval = 100
        else:
            display_interval = 1000 
            
        propagation(steps)


        #-----------------run fourrier transform and writing quantities in out files for each block----------------

        for nb in range(1, nblocks+1):

            fourier_transform(ReGtotsmp[nb], ImGtotsmp[nb], Nvalues, dwell, larmorfreq, tau, nsample, nb)

            xp.savetxt(f'density_{nb}.txt', denstot[nb])
            xp.savetxt(f'vacf_{nb}.txt', lattvacf[nb])
            xp.savetxt(f'vacfx_{nb}.txt', lattvacfx[nb])
            xp.savetxt(f'vacfy_{nb}.txt', lattvacfy[nb])
            xp.savetxt(f'vacfz_{nb}.txt', lattvacfz[nb])
            xp.savetxt(f'time_dep_Diffusion_coeff-D_over_D0_{nb}.txt', (sumvacf[nb]/(dim*1.0)/(a*a/(2.0*dim*dt))))
            #xp.savetxt('Signal.txt', xp.column_stack((ReGtot, ImGtot, magnitude, phase)))
            xp.savetxt(f'Signal_smp_{nb}.txt', xp.column_stack((ReGtotsmp[nb], ImGtotsmp[nb], magnitude[nb], phase[nb])))
            xp.savetxt(f'Diffusion_coeff_x_{nb}.txt', xp.column_stack((xsumvacf[nb]/dim*1.0, xsumvacfx[nb], xsumvacfy[nb], xsumvacfz[nb])))
            xp.savetxt(f'Diffusion_coeff_y_{nb}.txt', xp.column_stack((ysumvacf[nb]/dim*1.0, ysumvacfx[nb], ysumvacfy[nb], ysumvacfz[nb])))
            xp.savetxt(f'Diffusion_coeff_z_{nb}.txt', xp.column_stack((zsumvacf[nb]/dim*1.0, zsumvacfx[nb], zsumvacfy[nb], zsumvacfz[nb])))

            print(f"VACF Diffusion coefficient_{nb}: {sumvacf[nb, steps-1]/dim:.6e}")
            print(f"VACF Diffusion coefficient in direction x_{nb}: {sumvacfx[nb, steps-1]:.6e}")
            print(f"VACF Diffusion coefficient in direction y_{nb}: {sumvacfy[nb, steps-1]:.6e}")
            print(f"VACF Diffusion coefficient in direction z_{nb}: {sumvacfz[nb, steps-1]:.6e}")
            print("------------------------------------------------------------------------------")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The execution time is {elapsed_time:.2f} seconds.")
        exit()

    MPI.Finalize()

if __name__ == "__main__":
    main()
