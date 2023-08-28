#!/usr/bin/env python

import numpy as np
import pyscf
from pyscf import gto

# Setup the pyscf molecule
mol = pyscf.M(
    atom = [["H",0,0,0],["H",0,0,1.3794]],
    basis = "6-31g",
    spin = 0,
    symmetry = False,
    unit ="Bohr",
)

# Save the number of AOs, alpha electrons, and beta electrons.
with open("nelec.txt","w") as outfile:
    outfile.write("{:5d} {:5d} {:5d}\n".format(mol.nao, *mol.nelec))

with open("enuc.txt","w") as outfile:
    outfile.write("{: 20.16f}".format(mol.energy_nuc()))

# Save the one-electron hamiltonian
np.savetxt("ovlp.txt", mol.intor("int1e_ovlp"))
np.savetxt("oeis.txt", mol.intor("int1e_kin") + mol.intor("int1e_nuc"))

# Save the two-electron Coulomb repulsion
IIcc = np.reshape(mol.intor("int2e"),(mol.nao*mol.nao,mol.nao*mol.nao))
IIcc.tofile("teis.bin")
