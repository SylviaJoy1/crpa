from pyscf import df
import numpy as np
from pyscf import lib
einsum = lib.einsum

def kernel(crpa, screened = True):
    nmo = crpa.nmo
    nocc = crpa.nocc
    nvir = nmo - nocc
    e_mo_occ = crpa.mo_energy[:nocc] 
    e_mo_vir = crpa.mo_energy[nocc:] 
    
    canon_Lov, loc_Lpq = crpa.get_Lpq()
    
    if not screened:
        return einsum('Pij,Pkl->ijkl', loc_Lpq, loc_Lpq) 
        
    U = crpa.make_U()
    
    naux = np.shape(canon_Lov)[0]
    i_mat = np.zeros((naux,naux))
    
    U_occ = U[:nocc, :]
    U_vir = U[nocc:, :]
        
    for a in range(nvir):
        for i in range(nocc): 
            i_mat += (1-np.sum(np.abs(U_occ[i,:])**2)*np.sum(np.abs(U_vir[a,:])**2))*np.outer(canon_Lov[:,i,a]/(e_mo_occ[i] - e_mo_vir[a]), canon_Lov[:,i,a])
                
    i_tilde = np.linalg.inv(np.eye(naux)-4.0*i_mat)
        
    return einsum('Pij,PQ,Qkl->ijkl', loc_Lpq, i_tilde, loc_Lpq)

#Make the unitary transformation matrix from canonical to localized orbitals
def make_U(mf, loc_coeff=None):
    mol = mf.mol
    if loc_coeff is None:
        return np.eye(np.shape(mf.mo_coeff)[0]) #canonical to canonical
    # "C matrix stores the AO to localized orbital coefficients"
    #"Mole.intor() is provided to obtain the one- and two-electron AO integrals"
    S_atomic = mol.intor_symmetric('int1e_ovlp') #molecular
    # S_atomic = mol.pbc_intor('int1e_ovlp')
    U = np.dot(mf.mo_coeff.T, np.dot(S_atomic, loc_coeff))
    return U      

def get_Lpq(mf,  df_file, loc_coeff=None):
    if loc_coeff is None:
        loc_coeff = mf.mo_coeff
    nocc = np.count_nonzero(mf.mo_occ > 0)
    mol = mf.mol
    with_df = df.DF(mol)
    mf.with_df._cderi_to_save = df_file
    
    from pyscf.ao2mo import _ao2mo
    def _init_df_eris(mf, with_df, mo_coeff=None, norb=None):
        if norb is None:
            norb = len(mf.mo_occ)
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        #with_df = mf.with_df
        naux = with_df.get_naoaux()     
        Lpq_ = np.empty((naux, norb, norb))
        mo = np.asarray(mo_coeff, order='F')
        ijslice = (0, norb, 0, norb)
        p1 = 0
        Lpq = None
        for k, eri1 in enumerate(with_df.loop()):
            Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
            p0, p1 = p1, p1 + Lpq.shape[0]
            Lpq = Lpq.reshape(p1-p0, norb, norb)
            Lpq_[p0:p1] = Lpq[:,::]
        Lpq = None
        return Lpq_
    
    print('nocc', nocc)
    canon_Lov = _init_df_eris(mf, with_df)[:, :nocc, nocc:] #can probably specify norb and modify mo_coeff to only give ov section
    norb = np.shape(loc_coeff)[1]
    print(norb)
    loc_Lpq = _init_df_eris(mf, with_df, mo_coeff = loc_coeff, norb = norb)
    print('shape of canon lov', np.shape(canon_Lov))
    print('shape of loc lpq', np.shape(loc_Lpq))
    # canon_Lov = einsum('Lpq, pn, qm->Lnm', Lpq, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]) 
    # loc_Lpq = einsum('Lpq, pn, qm->Lnm', Lpq, loc_coeff, loc_coeff) 
    
    return canon_Lov, loc_Lpq
   
#TODO: include ROKS (wrote it already somewhere)
class cRPA(lib.StreamObject):
    def __init__(self, mf, df_file, loc_coeff=None):
        self.mf = mf
        self.mol = mf.mol
        # self.chkfile    =   mf.chkfile
        self.df_file = df_file
        
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
        self.nao        =   mf.mol.nao_nr()
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
        self.ERIs = kernel(self, screened)
        return self.ERIs

#def h1e_for_cas(casci, mo_coeff=None, ncas=None, ncore=None):
#    from functools import reduce
#    '''CAS space one-electron hamiltonian with DFT h1e (no double counting correction)
#    Args:
#        casci: a CASSCF/CASCI object or RHF object
#
#    Returns:
#        A tuple, the first is the effective one-electron DFT hamiltonian defined in CAS space,
#        the second is the electronic energy from core.
#    See also: pyscf/mcscf/casci.py
#    '''
#    if mo_coeff is None: mo_coeff = casci.mo_coeff
#    if ncas is None: ncas = casci.ncas
#    if ncore is None: ncore = casci.ncore
#    mo_core = mo_coeff[:,:ncore]
#    mo_cas = mo_coeff[:,ncore:ncore+ncas]
#
#    hcore = casci.get_hcore()
#    veff = casci._scf.get_veff()
#    energy_core = casci.energy_nuc()
#    #:if mo_core.size == 0:
#    #:    corevhf = 0
#    #:else:
#    #:    core_dm = numpy.dot(mo_core, mo_core.conj().T) * 2
#    #:    corevhf = casci.get_veff(casci.mol, core_dm)
#    #:    energy_core += numpy.einsum('ij,ji', core_dm, hcore).real
#    #:    energy_core += numpy.einsum('ij,ji', core_dm, corevhf).real * .5
#    #:h1eff = reduce(numpy.dot, (mo_cas.conj().T, hcore+corevhf, mo_cas))
#    h1eff = reduce(np.dot, (mo_cas.conj().T, hcore+veff, mo_cas))
#
#    # The core energy is meaningless for now, but doesn't matter for excitation energies
#    return h1eff, energy_core

from pyscf import mcscf
class cRPA_CASCI(mcscf.casci.CASCI):
    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, screened_ERIs=None):
        super().__init__(mf_or_mol, ncas, nelecas, ncore=None)
        self.screened_ERIs = screened_ERIs
        self.mf_or_mol = mf_or_mol
        
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
    #if it is not RHF-based, we have to modify the h1 used in CASCI
    
#    def get_veff(self, mol=None, dm=None, hermi=1):
#        if mol is None: mol = self.mol
#        if dm is None:
#            mocore = self.mo_coeff[:,:self.ncore]
#            dm = numpy.dot(mocore, mocore.conj().T) * 2
#        # use get_veff even if _scf is a DFT object
#        return self._scf.get_veff(mol, dm, hermi=hermi)
#
#    get_h1cas = h1e_for_cas  = h1e_for_cas

#    def get_h1eff(self, mo_coeff=None, ncas=None, ncore=None):
#        return self.h1e_for_cas(mo_coeff, ncas, ncore)
            
if __name__ == '__main__':
    from pyscf import gto, scf, dft
    from pyscf.mcscf import avas
    from pyscf.tools import molden
    from pyscf.tools import mo_mapping

    mol = gto.Mole()

    #ferrocene
    mol.atom = '''Fe 0.000000 0.000000 0.000000 
    C -0.713500 -0.982049 -1.648000 
    C 0.713500 -0.982049 -1.648000 
    C 1.154467 0.375109 -1.648000 
    C 0.000000 1.213879 -1.648000 
    C -1.154467 0.375109 -1.648000 
    H -1.347694 -1.854942 -1.638208 
    H 1.347694 -1.854942 -1.638208 
    H 2.180615 0.708525 -1.638208 
    H 0.000000 2.292835 -1.638208 
    H -2.180615 0.708525 -1.638208 
    C -0.713500 -0.982049 1.648000 
    C -1.154467 0.375109 1.648000 
    C -0.000000 1.213879 1.648000 
    C 1.154467 0.375109 1.648000 
    C 0.713500 -0.982049 1.648000 
    H -1.347694 -1.854942 1.638208 
    H -2.180615 0.708525 1.638208 
    H 0.000000 2.292835 1.638208 
    H 2.180615 0.708525 1.638208 
    H 1.347694 -1.854942 1.638208
    '''
    
    mol.basis = 'cc-pvtz-dk'
    mol.build()
    
    mf = scf.RHF(mol).x2c().density_fit()#.x2c() #should be ROHF for ferrocene, but RHF works
    xc_f = 'rhf'
    dm = mf.from_chk('ferrocene_rks.chk')
    #mf.chkfile = 'ferrocene_{}_x2c.chk'.format(xc_f)
    
    #mf.with_df = df.DF(mol)
    mf.with_df._cderi_to_save = 'ferrocene_{}_x2c.h5'.format(xc_f)
    
    
    #mf.kernel()
    mf.kernel(dm)
    
    nocc = mol.nelectron//2
    nmo = mf.mo_energy.size
    nvir = nmo - nocc
    print('number of electrons', mol.nelectron)
    print('nocc', nocc)
    
    ao_labels = ['Fe 3d', 'C 2pz']
    #ao_labels = ['Fe 3d']
    avas_obj = avas.AVAS(mf, ao_labels, threshold = 0.1, canonicalize=False) #orbs = coefficients of AO->localized orbs
    norb, ne_act, orbs = avas_obj.kernel()
    print(avas_obj.occ_weights)
    print(avas_obj.vir_weights)
    ncas = avas_obj.ncas
    nelecas = avas_obj.nelecas
    print(ncas)
    print(nelecas)
    ncore = avas_obj.ncore
    
    nocc_act = np.count_nonzero(avas_obj.occ_weights > 0.1)
    nvir_act = np.count_nonzero(avas_obj.vir_weights > 0.1)
    print('nocc_act', nocc_act, 'nvir_act', nvir_act)
        
    orbs = lib.chkfile.dump('ferrocene_{}_x2c.chk'.format(xc_f), 'orbs', orbs)
    ncas = lib.chkfile.dump('ferrocene_{}_x2c.chk'.format(xc_f), 'ncas', ncas)
    nelecas = lib.chkfile.dump('ferrocene_{}_x2c.chk'.format(xc_f), 'nelecas', nelecas)
                
    orbs = lib.chkfile.load('ferrocene_{}_x2c.chk'.format(xc_f), 'orbs')
    ncas = lib.chkfile.load('ferrocene_{}_x2c.chk'.format(xc_f), 'ncas')
    nelecas = lib.chkfile.load('ferrocene_{}_x2c.chk'.format(xc_f), 'nelecas')
    
    C_loc = orbs[:,nocc-nocc_act:nocc+nvir_act] 
                
    mycRPA = cRPA(mf, 'ferrocene_{}_x2c.h5'.format(xc_f), loc_coeff = C_loc)
    my_unscreened_eris = mycRPA.kernel(screened = False)
    print('ferrocene unscreened ERIs (eV)', [my_unscreened_eris[i,i,i,i]*27.2114 for i in range(ncas)])
    my_screened_eris = mycRPA.kernel(screened = True)
    print('ferrocene screened ERIs (eV)', [my_screened_eris[i,i,i,i]*27.2114 for i in range(ncas)])
                
    from pyscf import mcscf, fci
             
    print('cRPA_CASCI, unscreened U')
    mycas = cRPA_CASCI(mf, ncas, nelecas, screened_ERIs = my_unscreened_eris)
    mycas.canonicalization = False
    mycas.fcisolver.nroots = 15
    mycas.kernel(orbs)
                
    print('cRPA_CASCI, screened U')
    mycas = cRPA_CASCI(mf, ncas, nelecas, screened_ERIs = my_screened_eris)
    mycas.canonicalization = False
    mycas.fcisolver.nroots = 15
    mycas.kernel(orbs)
    
    print('conventional CASCI')
    mycas = mcscf.CASCI(mf, ncas, nelecas)
    mycas.fcisolver.nroots = 15
    mycas.kernel(orbs)

    print('DF-CASCI')
    mycas = mcscf.DFCASCI(mf, ncas, nelecas)
    mycas.canonicalization = False
    # mycas.verbose = 6
    # orbs = np.hstack( ( np.hstack( (mf.mo_coeff[:, :nocc-1], C_loc) ), mf.mo_coeff[:, nocc+1:] ) )
    mycas.fcisolver.nroots = 15
    #print('DFCASCI unscreened U', mycas.ao2mo(orbs))
    # print('DFCASCI t', mycas.h1e_for_cas(orbs, ncas))
    mycas.kernel(orbs)
