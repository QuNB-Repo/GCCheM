import numpy as np
from pygnme import wick, utils, slater, owndata

# NOTE: Throughout, we need use owndata(...) to convert numpy arrays to a
#       suitable format for the PyGNME code

# Load number of basis functions (nbsf), alpha (na), and beta (nb) electrons
(nbsf, na, nb) = map(int, np.genfromtxt('nelec.txt'))

# NOTE: It is possible that the number of MOs might not equal the number of basis 
#       functions, if the AOs are linearly dependent
nmo = nbsf

# Nuclear repulsion
enuc  = np.genfromtxt('enuc.txt')
# 1-electron hamiltonian matrix
hcore = owndata(np.genfromtxt('oeis.txt'))
# AO overlap matrix
ovlp  = owndata(np.genfromtxt('ovlp.txt'))
# 2-electron integrals in the form (pq|rs) -> [p*nbsf+q, r*nbsf+s]
eri   = owndata(np.reshape(np.fromfile('teis.bin'), (nbsf**2, nbsf**2)))

# Setup matrix builder object
# Arguments are Nbsf, Nmo, Nalpha, Nbeta, Overlap, Enuc
mb = slater.slater_uscf[float, float, float](nbsf, nbsf, na, nb, ovlp, enuc) 
# Add one- and two-body contributions
mb.add_one_body(hcore)
mb.add_two_body(eri)

# Load alpha coefficients, list [(C1a,C1b),(C2a,C2b),...]
# NOTE: You can structure this however you like!
C = [(np.genfromtxt('0.00_1.3794_a.txt'), np.genfromtxt('0.00_1.3794_b.txt')), 
     (np.genfromtxt('0.20_1.3794_a.txt'), np.genfromtxt('0.20_1.3794_b.txt')),
     (np.genfromtxt('0.20_1.3794_b.txt'), np.genfromtxt('0.20_1.3794_a.txt')) # This is the spin-flip determinant
    ]

# Compute coupling terms
nstate = len(C)
h = np.zeros((nstate, nstate))
s = np.zeros((nstate, nstate))
for x in range(nstate):
    for w in range(x, nstate):
        # Access occupied orbitals with CARMA data ownership
        Cxa, Cxb = owndata(C[x][0][:,:na]), owndata(C[x][1][:,:nb])
        Cwa, Cwb = owndata(C[w][0][:,:na]), owndata(C[w][1][:,:nb])

        # Compute the Hamiltonian and overlap matrix elements
        s[x,w], h[x,w] = mb.evaluate(Cxa, Cxb, Cwa, Cwb)
         
        # TODO: Need to get access to the <S^2> matrix elements

        # Save hermitian copy
        s[w,x] = np.conj(s[x,w])
        h[w,x] = np.conj(h[x,w])

print("NOCI Hamiltonian")
print(h)

print("\nNOCI Overlap")
print(s)

tmp, eigvec = utils.gen_eig_sym(nstate, h, s, thresh=1e-8)
eigval = tmp[0]

print("\nNOCI eigenvalues")
print(eigval)

print("\nNOCI eigenvectors")
print(eigvec)
