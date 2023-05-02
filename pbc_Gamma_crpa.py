from pyscf.pbc import df
import numpy as np
from pyscf import lib
einsum = lib.einsum
from pyscf.pbc.tools.pbc import madelung

def kernel(crpa, M=0, screened = True):
    nmo = crpa.nmo
    nocc = crpa.nocc
    nvir = nmo - nocc
    e_mo_occ = crpa.mo_energy[:nocc] 
    e_mo_vir = crpa.mo_energy[nocc:] 
    
    canon_Lov, loc_Lpq = crpa.get_Lpq()
    
    if not screened:
        return einsum('Pij,Pkl->ijkl', loc_Lpq, loc_Lpq)+M 
        
    U = crpa.make_U()
    
    naux = np.shape(canon_Lov)[0]
    i_mat = np.zeros((naux,naux))
    
    U_occ = U[:nocc, :]
    U_vir = U[nocc:, :]
        
    for a in range(nvir):
        for i in range(nocc): 
            i_mat += (1-np.sum(np.abs(U_occ[i,:])**2)*np.sum(np.abs(U_vir[a,:])**2))*np.outer(canon_Lov[:,i,a]/(e_mo_occ[i] - e_mo_vir[a]), canon_Lov[:,i,a])
                
    i_tilde = np.linalg.inv(np.eye(naux)-4.0*i_mat)
       
    scr_U = einsum('Pij,PQ,Qkl->ijkl', loc_Lpq, i_tilde, loc_Lpq)
    scr_M = M*einsum('Pij,PQ,Qkl->ijkl', loc_Lpq, i_tilde, loc_Lpq)/('Pij,Pkl->ijkl', loc_Lpq, loc_Lpq)
    return scr_U + scr_M

#Make the unitary transformation matrix from canonical to localized orbitals
def make_U(mf, loc_coeff=None):
    cell = mf.cell
    if loc_coeff is None:
        return np.eye(np.shape(mf.mo_coeff)[0]) #canonical to canonical
    # "C matrix stores the AO to localized orbital coefficients"
    #"Mole.intor() is provided to obtain the one- and two-electron AO integrals"
    # S_atomic = cell.intor_symmetric('int1e_ovlp')
    S_atomic = cell.pbc_intor('int1e_ovlp')
    U = np.dot(mf.mo_coeff.T, np.dot(S_atomic, loc_coeff))
    return U      

def get_Lpq(mf,  df_file, loc_coeff=None):
    if loc_coeff is None:
        loc_coeff = mf.mo_coeff
    cell = mf.cell
    a_gdf = df.GDF(cell)
    a_gdf._cderi = df_file
    naux = a_gdf.get_naoaux()
    nao = cell.nao
    nocc = np.count_nonzero(mf.mo_occ > 0)
    Lpq = np.empty((naux,nao,nao))
    kpt = np.zeros(3)
    p1 = 0
    for LpqR, LpqI, x in a_gdf.sr_loop((kpt,kpt), compact=False):
        p0, p1 = p1, p1 + LpqR.shape[0]
        Lpq[p0:p1] = LpqR.reshape(-1,nao,nao)
    
    canon_Lov = einsum('Lpq, pn, qm->Lnm', Lpq, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]) 
    loc_Lpq = einsum('Lpq, pn, qm->Lnm', Lpq, loc_coeff, loc_coeff) 
    
    return canon_Lov, loc_Lpq
   
#TODO: include ROKS, molecular code
class cRPA(lib.StreamObject):
    def __init__(self, mf, df_file, loc_coeff=None):
        self.mf = mf
        self.cell = mf.cell
        # self.chkfile    =   mf.chkfile
        self.df_file = df_file
        self.M = madelung(mf.cell)
        self.mo_coeff   =   mf.mo_coeff
        
        if loc_coeff is None:
            loc_coeff = self.mo_coeff
        self.loc_coeff = loc_coeff
    
        self.verbose    =   mf.verbose
        self.stdout     =   mf.stdout
        self.max_memory =   mf.max_memory
        
        ##################################################
        # don't modify the following attributes, they are not input options
        self.ERIs = None
        self.mo_energy  =   mf.mo_energy
        self.mo_occ     =   mf.mo_occ
        self.nao        =   mf.cell.nao_nr()
        self.nmo        =   len(mf.mo_occ)
        self.nocc       =   np.count_nonzero(mf.mo_occ > 0)
        self.nvir       =   self.nmo - self.nocc
    
    def get_Lpq(self):
        canon_Lov, loc_Lpq = get_Lpq(self.mf, self.df_file, self.loc_coeff)
        return canon_Lov, loc_Lpq
    
    def make_U(self):
        U = make_U(self.mf, self.loc_coeff)
        return U
    
    def kernel(self, screened = True):
        self.ERIs = kernel(self, M, screened)
        return self.ERIs

def h1e_for_cas(casci, mo_coeff=None, ncas=None, ncore=None):
    from functools import reduce
    '''CAS space one-electron hamiltonian with DFT h1e (no double counting correction)
    Args:
        casci: a CASSCF/CASCI object or RHF object
    
    Returns:
        A tuple, the first is the effective one-electron DFT hamiltonian defined in CAS space,
        the second is the electronic energy from core.
    See also: pyscf/mcscf/casci.py
    '''
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    if ncas is None: ncas = casci.ncas
    if ncore is None: ncore = casci.ncore
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:ncore+ncas]
    
    hcore = casci.get_hcore()
    veff = casci._scf.get_veff()
    energy_core = casci.energy_nuc()
    #:if mo_core.size == 0:
    #:    corevhf = 0
    #:else:
    #:    core_dm = numpy.dot(mo_core, mo_core.conj().T) * 2
    #:    corevhf = casci.get_veff(casci.mol, core_dm)
    #:    energy_core += numpy.einsum('ij,ji', core_dm, hcore).real
    #:    energy_core += numpy.einsum('ij,ji', core_dm, corevhf).real * .5
    #:h1eff = reduce(numpy.dot, (mo_cas.conj().T, hcore+corevhf, mo_cas))
    h1eff = reduce(np.dot, (mo_cas.conj().T, hcore+veff, mo_cas))

    # The core energy is meaningless for now, but doesn't matter for excitation energies
    return h1eff, energy_core
        
from pyscf import mcscf
class cRPA_CASCI(mcscf.casci.CASCI):
    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, screened_ERIs=None):
        super().__init__(mf_or_mol, ncas, nelecas, ncore=None)
        self.screened_ERIs = screened_ERIs
        
    def get_h2eff(self, mo_coeff=None):
        '''Compute the active space two-particle Hamiltonian.

        Note It is different to get_h2cas when df.approx_hessian is applied.
        in which get_h2eff function returns the DF integrals while get_h2cas
        returns the regular 2-electron integrals.
        '''
        if self.screened_ERIs is None:
            return self.ao2mo(mo_coeff)
            #this is not using density fitting.
        return self.screened_ERIs

    # def get_h1eff(self, mo_coeff=None, ncas=None, ncore=None):
    #     return self.h1e_for_cas(mo_coeff, ncas, ncore)
    
    def get_veff(self, mol=None, dm=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None:
            mocore = self.mo_coeff[:,:self.ncore]
            dm = numpy.dot(mocore, mocore.conj().T) * 2
        # use get_veff even if _scf is a DFT object
        return self._scf.get_veff(mol, dm, hermi=hermi)

    get_h1cas = h1e_for_cas = h1e_for_cas

    def get_h1eff(self, mo_coeff=None, ncas=None, ncore=None):
        return self.h1e_for_cas(mo_coeff, ncas, ncore)
            
if __name__ == '__main__':
    
    #hBN+C2
    from pyscf.pbc import gto, dft
    from ase.lattice.hexagonal import Graphene 
    
    import multiprocessing
    from multiprocessing import Pool
    print('number of logical cores', multiprocessing.cpu_count())
    
    def task(i):
        with open('madelung_screened_dzv_{}.txt'.format(i), 'w') as scr:
            with open('madelung_unscreened_dzv_{}.txt'.format(i), 'w') as uscr:
                nx = i
                ny = i
               
                cell = gto.Cell()
                cell.unit = 'b'
                atoms = open("atoms_{}.txt".format(i), "r")
                cell_abc = open("cell_{}.txt".format(i), "r")
                cell.atom = atoms.read()
                cell.a = cell_abc.read()
                
                cell.basis = 'gth-dzv'
                cell.pseudo = 'gth-pade'
                cell.ke_cutoff = 80.
                cell.verbose = 5
                cell.build()
                
                mf = dft.RKS(cell).density_fit()
                mf.xc = 'pbe'
                mf.with_df._cderi_to_save = 'hbn_c2_dzv_{}.h5'.format(nx)
                # mf.with_df._cderi = 'hbn_c2_dzv_{}.h5'.format(nx)
                # mf.chkfile = 'hbn_c2_dzv_{}.chk'.format(nx)
                # dm = mf.from_chk('hbn_c2_dzv_{}.chk'.format(nx))
                mf.chkfile = 'hbn_c2_dzv_{}.chk'.format(nx)
                 
                # mf.kernel(dm)
                mf.kernel()
                
                # from pyscf.pbc.scf.chkfile import load_scf
                # cell, scf_res = load_scf('hbn_c2_gdf_{}x{}x1.chk'.format(nx,ny))
                # cell.verbose = 6
                # mf = dft.RKS(cell).density_fit()
                # mf.with_df._cderi = 'hbn_c2_pbc_gdf_{}x{}x1.h5'.format(nx,ny)
                # mf.xc = 'pbe'
                # mf.mo_coeff = scf_res['mo_coeff']
                # mf.mo_energy = scf_res['mo_energy']
                # mf.mo_occ = scf_res['mo_occ']
                # mf.e_tot = scf_res['e_tot']
                # mf.converged = True
                # energy_nuc = mf.energy_nuc()
                
                nocc = cell.nelectron//2
                nmo = mf.mo_energy.size
                nvir = nmo - nocc
                print('HOMO E: ', mf.mo_energy[nocc-1], 'LUMO E: ', mf.mo_energy[nocc])
                
                # Using P-M to mix and localize the HOMO/LUMO
                from pyscf import lo
                idcs = np.ix_(np.arange(nmo), [nocc-1, nocc])
                mo_init = lo.PM(cell, mf.mo_coeff[idcs])
                C_loc = mo_init.kernel()
                lib.chkfile.dump('hbn_c2_dzv_{}.chk'.format(nx), 'C_loc', C_loc)
                
                C_loc = lib.chkfile.load('hbn_c2_dzv_{}.chk'.format(nx), 'C_loc')
                
                mycRPA = cRPA(mf, 'hbn_c2_dzv_{}.h5'.format(nx), loc_coeff = C_loc)
                my_unscreened_eris = mycRPA.kernel(screened = False)
                my_screened_eris = mycRPA.kernel(screened = True)
                # print('hBN+C2 C2pz unscreened ERIs (eV)', my_unscreened_eris*27.2114)
                # print('hBN+C2 C2pz screened ERIs (eV)', my_screened_eris*27.2114)
                
                from pyscf import mcscf, fci
             
                uscr.write('cRPA_CASCI, unscreened U \n')
                ncas  = 2
                ne_act = 2
                mycas = cRPA_CASCI(mf, ncas, ne_act, screened_ERIs = my_unscreened_eris)
                mycas.canonicalization = False
                mycas.verbose = 6
                orbs = np.hstack( ( np.hstack( (mf.mo_coeff[:, :nocc-1], C_loc) ), mf.mo_coeff[:, nocc+1:] ) )
                mycas.fcisolver.nroots = 4
                mycas.kernel(orbs)
                uscr.write('t={} \n'.format(mycas.get_h1eff()))
                uscr.write('n={} Exc={} U={}'.format(nx, mycas.e_tot, (my_unscreened_eris[0,0,0,0]+my_unscreened_eris[1,1,1,1])/2))
                uscr.write('\n')
                uscr.write('unscreened eris \n')
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            for l in range(2):
                                uscr.write('({}{}|{}{})={} \n'.format(i,j,k,l, my_unscreened_eris[i,j,k,l]*27.2114))
                for exc in mycas.e_tot:
                    uscr.write('{}'.format(27.2114*(exc-mycas.e_tot[0])))
                    uscr.write('\n')
                
                scr.write('cRPA_CASCI, screened U \n')
                mycas = cRPA_CASCI(mf, ncas, ne_act, screened_ERIs = my_screened_eris)
                mycas.canonicalization = False
                mycas.verbose = 6
                # orbs = np.hstack( ( np.hstack( (mf.mo_coeff[:, :nocc-1], C_loc) ), mf.mo_coeff[:, nocc+1:] ) )
                mycas.fcisolver.nroots = 4
                mycas.kernel(orbs)
                scr.write('t={} \n'.format(mycas.get_h1eff()))
                scr.write('n={} Exc={} U={}'.format(nx, mycas.e_tot, (my_screened_eris[0,0,0,0]+my_screened_eris[1,1,1,1])/2))
                scr.write('\n')
                scr.write('screened eris \n')
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            for l in range(2):
                                scr.write('({}{}|{}{})={} \n'.format(i,j,k,l, my_screened_eris[i,j,k,l]*27.2114))
                for exc in mycas.e_tot[1:]:
                    scr.write('{}'.format(27.2114*(exc-mycas.e_tot[0])))
                    scr.write('\n')
            
        return None    
    
    # create the process pool
    with Pool() as pool:
        # call the same function with different data in parallel
        for result in pool.map(task, range(2,6)):
            # report the value to show progress
            print(result)
