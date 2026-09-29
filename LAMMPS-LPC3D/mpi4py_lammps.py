from mpi4py import MPI
from lammps import lammps
import os
import shutil
import subprocess
import re
import glob
import numpy as np
from pathlib import Path

POS_FILE = Path("CM_positions.xyz")
BOX_FILE = Path("data.lmp")


boxes = [(35.0 ,65.0 ,21.0 ,50.0), (35.0 ,65.0 ,100.0 ,129.0)]

def compute_densities(sub_comm, pore_size, xyz_file, boxes, species=("M1","M2","M3")):
    if sub_comm.Get_rank() != 0:
        return None
    
    xmin = 50 - pore_size / 2
    xmax = 50 + pore_size / 2

    nframes = 0
    totals = { (b,sp): 0 for b in range(len(boxes)) for sp in species }

    with open(xyz_file) as f:
        while True:
            line = f.readline()
            if not line: break
            n = int(line.strip())
            f.readline()
            frame = [f.readline().split() for _ in range(n)]
            nframes += 1

            for b_idx, (ymin,ymax,zmin,zmax) in enumerate(boxes):
                for sp in species:
                    count = sum(
                        xmin<=float(x)<=xmax and
                        ymin<=float(y)<=ymax and
                        zmin<=float(z)<=zmax
                        for lbl,x,y,z in frame if lbl.startswith(sp)
                    )
                    totals[(b_idx,sp)] += count

    results = []
    for b_idx, (ymin,ymax,zmin,zmax) in enumerate(boxes):
        vol = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
        for sp in species:
            dens = totals[(b_idx,sp)] / (vol * nframes)
            results.append((b_idx, sp, dens))
    return results

def _read_box_limits(data_file: Path):
    xmin = xmax = ymin = ymax = zmin = zmax = None
    with data_file.open() as f:
        for line in f:
            if "xlo" in line and "xhi" in line:
                xmin, xmax = map(float, line.split()[:2])
            elif "ylo" in line and "yhi" in line:
                ymin, ymax = map(float, line.split()[:2])
            elif "zlo" in line and "zhi" in line:
                zmin, zmax = map(float, line.split()[:2])
            if all(v is not None for v in (xmin, xmax, ymin, ymax, zmin, zmax)):
                break
    return xmin, xmax, ymin, ymax, zmin, zmax

def read_input(input_file="sites2cm.inpt"):
    with open(input_file, "r") as f:
        filetype  = int(f.readline())
        nconfigs  = int(f.readline())
        nmol      = int(f.readline())
        natom_max = int(f.readline())

        nionstot  = np.zeros(nmol, dtype=int)
        ntypes    = np.zeros(nmol, dtype=int)
        masstot   = np.zeros(nmol)
        masstype  = np.zeros((nmol, natom_max))

        nionstotmax = 0
        newnionstot = 0

        for i in range(nmol):
            ntypes[i]   = int(f.readline())
            nionstot[i] = int(f.readline())
            newnionstot += nionstot[i]
            for j in range(ntypes[i]):
                masstype[i, j] = float(f.readline())
                masstot[i]    += masstype[i, j]
            nionstotmax = max(nionstotmax, nionstot[i])

    xmin = xmax = ymin = ymax = zmin = zmax = None
    xmin, xmax, ymin, ymax, zmin, zmax = _read_box_limits(BOX_FILE)

    filein  = "positions.xyz"
    fileout = "CM_positions.xyz"

    return (
        filetype, nconfigs,
        nmol, natom_max,
        nionstot, ntypes,
        masstot, masstype,
        nionstotmax, newnionstot,
        xmin, xmax, ymin, ymax, zmin, zmax,
        xmax - xmin, ymax - ymin, zmax - zmin,
        filein, fileout
    )

def calculate_center_of_mass(
    filetype, nconfigs,
    nmol, natom_max,
    nionstot, ntypes,
    masstot, masstype,
    nionstotmax, newnionstot,
    xmin, xmax, ymin, ymax, zmin, zmax,
    Lx, Ly, Lz,
    filein, fileout,
):
    xsum = np.zeros(nionstotmax)
    ysum = np.zeros(nionstotmax)
    zsum = np.zeros(nionstotmax)
    x0   = np.zeros(nionstotmax)
    y0   = np.zeros(nionstotmax)
    z0   = np.zeros(nionstotmax)

    with open(filein, "r") as fin, open(fileout, "w") as fout:

        # disp.out  -------------------------------------------------
        if filetype == 1:
            fout.write(fin.readline())                     
            for i in range(nmol):                          
                for j in range(nionstot[i]):
                    fin.readline()                         
                    if (j + 1) % ntypes[i] == 0:
                        fout.write(f"{i + 1}\n")

        # loop configs -----------------------------------------------
        for k in range(nconfigs):
            if (k + 1) % 10 == 0:
                print(f"[center_of_masses] {k + 1}/{nconfigs} configurations")

            if filetype == 2:
                fin.readline()                             # natoms initial
                comment = fin.readline().rstrip("\n")
                fout.write(f"{newnionstot}\n{comment}\n")

            for i in range(nmol):
                xsum.fill(0.0); ysum.fill(0.0); zsum.fill(0.0)

                for l in range(nionstot[i]):
                    for j in range(ntypes[i]):

                        if filetype == 2:
                            spec, xs, ys, zs = fin.readline().split()
                        else:
                            xs, ys, zs = fin.readline().split()

                        x = float(xs) - xmin
                        y = float(ys) - ymin
                        z = float(zs) - zmin

                        if j == 0:  # pivot
                            x0[l], y0[l], z0[l] = x, y, z
                            xsum[l] = x * masstot[i]
                            ysum[l] = y * masstot[i]
                            zsum[l] = z * masstot[i]
                        else:
                            dx, dy, dz = x - x0[l], y - y0[l], z - z0[l]

                            if filetype != 1:              # PBC
                                if dx >  Lx / 2: dx -= Lx
                                if dx < -Lx / 2: dx += Lx
                                if dy >  Ly / 2: dy -= Ly
                                if dy < -Ly / 2: dy += Ly
                                if dz >  Lz / 2: dz -= Lz
                                if dz < -Lz / 2: dz += Lz

                            xsum[l] += dx * masstype[i, j]
                            ysum[l] += dy * masstype[i, j]
                            zsum[l] += dz * masstype[i, j]

                    xcm = xsum[l] / masstot[i]
                    ycm = ysum[l] / masstot[i]
                    zcm = zsum[l] / masstot[i]

                    if filetype != 1:                     
                        if xcm < 0:   xcm += Lx
                        if xcm > Lx:  xcm -= Lx
                        if ycm < 0:   ycm += Ly
                        if ycm > Ly:  ycm -= Ly
                        if zcm < 0:   zcm += Lz
                        if zcm > Lz:  zcm -= Lz

                    # Writing ---------------------------------------------
                    if filetype == 2:                      
                        label = f"M{i + 1}"
                        fout.write(
                            f"{label:<3s}"
                            f"{xcm + xmin:15.5f}"
                            f"{ycm + ymin:15.5f}"
                            f"{zcm + zmin:15.5f}\n"
                        )
                    elif filetype == 0:                    # positions
                        fout.write(
                            f"{xcm + xmin:15.5f}"
                            f"{ycm + ymin:15.5f}"
                            f"{zcm + zmin:15.5f}\n"
                        )
                    else:                                  # disp.out
                        fout.write(f"{xcm:15.5f}{ycm:15.5f}{zcm:15.5f}\n")

def check_num_simulations(num_simulations, size, rank):
    
    if num_simulations > size:
        if rank == 0:
            print(f"Error : Number of simulations ({num_simulations}) cannot be greater than the total number of MPI processes ({size}).")
        MPI.Finalize()
        exit()

def distribute_processes(num_simulations, size, rank):

    base_processes = size // num_simulations  
    extra_processes = size % num_simulations  

    if rank < extra_processes * (base_processes + 1):
        processes_per_simulation = base_processes + 1
        simulation_id = rank // processes_per_simulation
    else:
        processes_per_simulation = base_processes
        simulation_id = (rank - extra_processes) // processes_per_simulation

    return simulation_id, processes_per_simulation

def setup_simulation_folder(sub_comm, pore_size, zmat_files, zmat_values, inp_files):
   
    folder = f"simulation_{pore_size}"
    if sub_comm.Get_rank() == 0:
        os.makedirs(folder, exist_ok=True)

        fixed_files = ["CG1.xyz", "CG2.xyz", "il.ff", "fftool", "packmol", "sites2cm.inpt"]
        input_files = fixed_files + list(zmat_files) + list(inp_files)
        for file in input_files:
            if os.path.isfile(file):
                shutil.copy(file, folder)

        modify_positions(folder, pore_size)
        run_fftool_and_packmol(folder, zmat_files, zmat_values)

    sub_comm.Barrier() 
    os.chdir(folder)
    return folder

def modify_positions(folder, pore_size):

    shift_plus = pore_size / 2
    shift_minus = -pore_size / 2
    
    for file_names in ["CG1.xyz", "CG2.xyz"]:
        path = os.path.join(folder, file_names)
        if os.path.isfile(path):
            with open(path, "r") as f:
                lines = f.readlines()

            with open(path, "w") as f:
                f.write(lines[0])
                f.write(lines[1])

                for i in range(2, 2 + 388):
                    parts = lines[i].strip().split()
                    if len(parts) == 4:
                        atom, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                        x += shift_plus
                        f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")

                for i in range(len(lines) - 388, len(lines)):
                    parts = lines[i].strip().split()
                    if len(parts) == 4:
                        atom, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                        x += shift_minus
                        f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")

def run_fftool_and_packmol(folder, zmat_files, zmat_values):

    os.chdir(folder)

    cmd = ["./fftool"]
    for val, zf in zip(zmat_values, zmat_files):
        cmd += [str(val), str(zf)]
    cmd += ["1", "CG1.xyz", "1", "CG2.xyz", "-b", "100.0"]

    with open("fftool.log", "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=log)

    file_path = "pack.inp"

    old_box = "98.5000 98.5000 98.5000"
    new_box = "98.5000 98.5000 148.5000"

    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "structure CG1_pack.xyz" in line or "structure CG2_pack.xyz" in line:
            new_lines.append(line)
            new_lines.append(lines[i+1])
            new_lines.append("  fixed 0. 0. 0. 0. 0. 0.\n")
            i += 3
        elif "inside box" in line and old_box in line:
            new_lines.append(line.replace(old_box, new_box))
            i += 1
        else:
            new_lines.append(line)
            i += 1

    with open(file_path, 'w') as f:
        f.writelines(new_lines)

    with open("packmol.log", "w") as output_file:
        with open("pack.inp", "r") as pack_file:
            subprocess.run(["./packmol"], stdin=pack_file, stdout=output_file, stderr=subprocess.STDOUT, check=True)

    cmd.append("-l")
    with open("fftool.log1", "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=log)

    lmp_file = "data.lmp"
    with open(lmp_file, 'r') as f:
        lines = f.readlines()

    new_lines = [
        "0.000000 150.000000 zlo zhi\n" if "zlo zhi" in line else line
        for line in lines
    ]

    with open(lmp_file, 'w') as f:
        f.writelines(new_lines)

    os.chdir("..")

def run_lammps(sub_comm, file_lmp, use_gpu=False, ngpu=1):

    if use_gpu:
        cmdargs = ["-k", "on", "g", "1", "-sf", "kk"]
    else:
        cmdargs = []

    lmp = lammps(cmdargs=cmdargs, comm=sub_comm)
    lmp.command("processors * * *")
    lmp.file(file_lmp)
    sub_comm.Barrier() 

def nvt_simulation_setup(sub_comm, inp_files):
    nvt_folder = "NVT"
    if sub_comm.Get_rank() == 0:
        if not os.path.exists(nvt_folder):
            os.makedirs(nvt_folder)

        input_files_nvt = [ "sites2cm.inpt", inp_files[1]]
        for file in input_files_nvt:
            if os.path.isfile(file):
                shutil.copy(file, nvt_folder)
        
        data_files = glob.glob("data.*.lmp")
        for file in data_files:
            if re.match(r"data\.\d+\.lmp$", file): 
                target_path = os.path.join(nvt_folder, "data.lmp")
                shutil.copy(file, target_path)
                break

    sub_comm.Barrier() 
    os.chdir(nvt_folder)
    return nvt_folder

def center_of_masses(sub_comm, param_file: str = "sites2cm.inpt"):

    if sub_comm.Get_rank() == 0:
        params = read_input(param_file)        
        calculate_center_of_mass(*params)     

    sub_comm.Barrier()


def run_simulation_lammps(pore_sizes, zmat_files, zmat_values, inp_files,
                          comm, rank, size, use_gpu, ngpu):

    num_simulations = len(pore_sizes)

    check_num_simulations(num_simulations, size, rank)

    simulation_id, _ = distribute_processes(num_simulations, size, rank)
    simulation_id = int(simulation_id)
    sub_comm = comm.Split(simulation_id, rank)

    gpu_mode = bool(use_gpu and ngpu > 0)

    # ======= calculate the total number of GPUs across all nodes =======
    if gpu_mode:
        hostname = MPI.Get_processor_name()

        # Each rank sends (hostname, number of GPUs on this node)
        local_info = (hostname, ngpu)
        all_info = comm.allgather(local_info)

        # host -> number of GPUs on this host (take the maximum value seen for this host)
        host_ngpu = {}
        for h, n in all_info:
            if h not in host_ngpu or n > host_ngpu[h]:
                host_ngpu[h] = n

        # Total number of GPUs across the entire job (sum over all nodes)
        total_ngpu = sum(host_ngpu.values())
    else:
        total_ngpu = 0

    # One simulation per global GPU (across all nodes)
    sims_per_wave = min(total_ngpu, num_simulations) if gpu_mode else num_simulations

    results = [None] * num_simulations if rank == 0 else None

    for wave_start in range(0, num_simulations, sims_per_wave):
        wave_end = min(wave_start + sims_per_wave, num_simulations)

        active = (wave_start <= simulation_id < wave_end)
        send = None

        if active:
            sim_index = simulation_id
            pore_size = pore_sizes[sim_index]
    
            if gpu_mode:
                gpu_id = sim_index % ngpu
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            setup_simulation_folder(sub_comm, pore_size, zmat_files, zmat_values, inp_files)
            sub_comm.Barrier()

            run_lammps(sub_comm, str(inp_files[0]), gpu_mode, 1 if gpu_mode else 0)
            sub_comm.Barrier()

            nvt_simulation_setup(sub_comm, inp_files)
            sub_comm.Barrier()

            run_lammps(sub_comm, str(inp_files[1]), gpu_mode, 1 if gpu_mode else 0)
            sub_comm.Barrier()

            center_of_masses(sub_comm)
            densities = compute_densities(sub_comm, pore_size, POS_FILE, boxes)

            os.chdir("..")
            os.chdir("..")

            if sub_comm.Get_rank() == 0:
                send = (sim_index, pore_size, densities)

        items = comm.gather(send, root=0)

        if rank == 0:
            for item in items:
                if item is not None:
                    i, p, d = item
                    results[i] = (p, d)

        comm.Barrier()

    if rank == 0:
        order = [(0, 'M1'), (0, 'M2'), (0, 'M3'),
                 (1, 'M1'), (1, 'M2'), (1, 'M3')]

        dens_table = []
        for i in range(num_simulations):
            pore_size, dens = results[i]
            dct = {(b, s): v for b, s, v in dens}
            row = [pore_size] + [dct.get(k, 0.0) for k in order]
            dens_table.append(row)
    else:
        dens_table = None

    dens_table = comm.bcast(dens_table, root=0)

    tab__dens = [
        [
            [0.0, row[1], 0.0, row[4]],  # M1
            [0.0, row[2], 0.0, row[5]],  # M2
            [0.0, row[3], 0.0, row[6]],  # M3
        ]
        for row in dens_table
    ]

    return tab__dens


 
 



