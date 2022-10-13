from pyscf.pbc import df
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
    U = crpa.make_U()
    
    naux = np.shape(canon_Lov)[0]
    i_mat = np.zeros((naux,naux))
    
    U_occ = U[:nocc, :]
    U_vir = U[nocc:, :]
        
    for a in range(nvir):
        for i in range(nocc): 
            i_mat += (1-np.sum(np.abs(U_occ[i,:])**2)*np.sum(np.abs(U_vir[a,:])**2))*einsum('P,Q->PQ', canon_Lov[:,i,a]/(e_mo_occ[i] - e_mo_vir[a]), canon_Lov[:,i,a])
            # i_mat += (1-np.sum(np.abs(U_occ[i,:])**2)*np.sum(np.abs(U_vir[a,:])**2))*np.outer(Lov[:,i,a]/(e_mo_occ[i] - e_mo_vir[a]), Lov[:,i,a])
                
    i_tilde = np.linalg.inv(np.eye(naux)-4.0*i_mat)
         
    if screened is False:
        return einsum('Pij,Pkl->ijkl', loc_Lpq, loc_Lpq) 
        
    return einsum('Pij,PQ,Qkl->ijkl', loc_Lpq, i_tilde, loc_Lpq)

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
    
    def kernel(self):
        self.ERIs = kernel(self)
        return self.ERIs

#The DF-CASSCF class overwrote get_h2eff, get_veff, and get_jk of CASSCF
#https://pyscf.org/_modules/pyscf/mcscf/df.html#density_fit
# pyscf.mcscf.df.density_fit(casscf, auxbasis=None, with_df=None)
# Generate DF-CASSCF for given CASSCF object. It is done by overwriting three CASSCF member functions:
# casscf.ao2mo which generates MO integrals
# casscf.get_veff which generate JK from core density matrix
# casscf.get_jk 
#I do not need to edit get_veff, get_jk.
#You could have half-screened, half-active orbitals as part of h1eff
#h1eff_{ab} = \sum_{\lambda\sigma} P^{core}_{\lambda\sigma}(ab|\lambda\sigma) - 0.5 * \sum_{\lambda\sigma} P^{core}_{\lambda\sigma}(a\sigma|\lambda b)
#But surely we ignore this?
from pyscf import mcscf
class cRPA_CASCI(mcscf.casci.CASCI):
    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, screened_ERIs=None):
        super().__init__(mf_or_mol, ncas, nelecas, ncore=None)
        self.screened_ERIs = None
        
    def get_h2eff(self, mo_coeff=None):
        '''Compute the active space two-particle Hamiltonian.

        Note It is different to get_h2cas when df.approx_hessian is applied.
        in which get_h2eff function returns the DF integrals while get_h2cas
        returns the regular 2-electron integrals.
        '''
        if self.screened_ERIs is None:
            return self.ao2mo(mo_coeff)
        return self.screened_ERIs

if __name__ == '__main__':
    #hBN+C2
    from pyscf.pbc import gto, dft
    cell = gto.Cell()
    cell.atom='''
    C             0.0000000000        0.0022633259        0.0000000000
    C             0.0000000000        1.3753623410        0.0000000000
    B             2.4877331551        0.0126249220        0.0000000000
    N             2.5260080244        1.4435321178        0.0000000000
    B             4.9958402174       -0.0023392202        0.0000000000
    N             5.0069039095        1.4401349492        0.0000000000
    B             7.5041597826       -0.0023392202        0.0000000000
    N             7.4930960905        1.4401349492        0.0000000000
    B            10.0122668449        0.0126249220        0.0000000000
    N             9.9739919756        1.4435321178        0.0000000000
    B            -1.2773800442        2.1590715924        0.0000000000
    N            -1.2591188862        3.6100329461        0.0000000000
    B             1.2773800442        2.1590715924        0.0000000000
    N             1.2591188862        3.6100329461        0.0000000000
    B             3.7663875637        2.1673282174        0.0000000000
    N             3.7550194127        3.6103930256        0.0000000000
    B             6.2500000000        2.1647177630        0.0000000000
    N             6.2500000000        3.6089321229        0.0000000000
    B             8.7336124363        2.1673282174        0.0000000000
    N             8.7449805873        3.6103930256        0.0000000000
    B            -2.5015923324        4.3329533228        0.0000000000
    N            -2.5043241319        5.7797523707        0.0000000000
    B            -0.0000000000        4.3206914723        0.0000000000
    N             0.0000000000        5.7716115082        0.0000000000
    B             2.5015923324        4.3329533228        0.0000000000
    N             2.5043241319        5.7797523707        0.0000000000
    B             5.0008155894        4.3309779705        0.0000000000
    N             5.0004133642        5.7756087394        0.0000000000
    B             7.4991844106        4.3309779705        0.0000000000
    N             7.4995866358        5.7756087394        0.0000000000
    B            -3.7526337581        6.5004832687        0.0000000000
    N            -3.7584315874        7.9493931575        0.0000000000
    B            -1.2498755049        6.4943568803        0.0000000000
    N            -1.2524702829        7.9400325834        0.0000000000
    B             1.2498755049        6.4943568803        0.0000000000
    N             1.2524702829        7.9400325834        0.0000000000
    B             3.7526337581        6.5004832687        0.0000000000
    N             3.7584315874        7.9493931575        0.0000000000
    B             6.2500000000        6.4893393890        0.0000000000
    N             6.2500000000        7.9338681655        0.0000000000
    B            -5.0121775306        8.6591059371        0.0000000000
    N            -5.0426492007       10.1120768419        0.0000000000
    B            -2.5024068179        8.6625769188        0.0000000000
    N            -2.5142169182       10.1036473994        0.0000000000
    B             0.0000000000        8.6590020733        0.0000000000
    N            -0.0000000000       10.1014666974        0.0000000000
    B             2.5024068179        8.6625769188        0.0000000000
    N             2.5142169182       10.1036473994        0.0000000000
    B             5.0121775306        8.6591059371        0.0000000000
    N             5.0426492007       10.1120768419        0.0000000000
    '''
    cell.a = '''
    12.5 0.0 0.0
      -6.249999999999998 10.825317547305485 0.0
      0.0 0.0 20.0'''
    cell.unit = 'A'
    
    cell.basis = 'gth-dzv'
    cell.pseudo = 'gth-pade'
    cell.ke_cutoff = 100
    cell.exp_to_discard=0.1
    cell.verbose = 6
    # cell.dimension = 2 
    cell.build()
    
    # mf = dft.RKS(cell).density_fit()
    # mf.xc = 'pbe'
    # mf.with_df._cderi_to_save = 'hbn_c2_pbc_gdf.h5'
    # mf.chkfile = 'hbn_c2.chk'
    # dm = mf.from_chk('hbn_c2.chk')
    
    # mf.kernel(dm)
    # mf.kernel()
    
    from pyscf.pbc.scf.chkfile import load_scf
    cell, scf_res = load_scf('hbn_c2.chk')
    cell.verbose = 6
    mf = dft.RKS(cell).density_fit()
    mf.xc = 'pbe'
    mf.mo_coeff = scf_res['mo_coeff']
    mf.mo_energy = scf_res['mo_energy']
    mf.mo_occ = scf_res['mo_occ']
    mf.e_tot = scf_res['e_tot']
    mf.converged = True
    energy_nuc = mf.energy_nuc()
    
    nocc = cell.nelectron//2
    nmo = mf.mo_energy.size
    nvir = nmo - nocc
    print('HOMO E: ', mf.mo_energy[nocc-1], 'LUMO E: ', mf.mo_energy[nocc])
    
    # Using P-M to mix and localize the HOMO/LUMO
    # from pyscf import lo
    # idcs = np.ix_(np.arange(nmo), [nocc-1, nocc])
    # mo_init = lo.PM(cell, mf.mo_coeff[idcs])
    # C_loc = mo_init.kernel()
    # lib.chkfile.dump('hbn_c2.chk', 'C_loc', C_loc)
    
    C_loc = lib.chkfile.load('hbn_c2.chk', 'C_loc')
    
    mycRPA = cRPA(mf, 'hbn_c2_pbc_gdf.h5', loc_coeff = C_loc)
    # my_unscreened_eris = mycRPA.kernel(screened = False)
    # print('hBN+C2 C2pz unscreened ERIs (eV)', my_unscreened_eris*27.2114)
    my_screened_eris = mycRPA.kernel()
    print('hBN+C2 C2pz screened ERIs (eV)', my_screened_eris*27.2114)
    
    from pyscf import mcscf, fci
    # weights = np.ones(4)/4
    # solver1 = fci.addons.fix_spin(fci.direct_spin1.FCI(cell), ss=2) # <S^2> = 2 for Triplet
    # solver1.spin = 2
    # solver1.nroots = 1
    # solver2 = fci.addons.fix_spin(fci.direct_spin1.FCI(cell), ss=0) # <S^2> = 0 for Singlet
    # solver2.spin = 0
    # solver2.nroots = 3

    ncas  = 2
    ne_act = 2
    mycas = cRPA_CASCI(mf, ncas, ne_act, screened_ERIs = my_screened_eris)
    mycas.canonicalization = False
    # mcscf.state_average_mix_(mycas, [solver1, solver2], weights)
    mycas.verbose = 6
    orbs = np.hstack( ( np.hstack( (mf.mo_coeff[:, :nocc-1], C_loc) ), mf.mo_coeff[:, nocc+1:] ) )
    mycas.fcisolver.nroots = 4
    mycas.kernel(orbs)
  
    h1, ecore = mycas.get_h1eff(orbs)
    print('cRPA_CASCI h1', h1)
    print('diagonalized cRPA_CASCI h1', np.linalg.eigh(h1))









    
    print('Now for the custom H(h1, h2)...')
    # #https://pyscf.org/user/ci.html
    from functools import reduce  
    ncore = nocc-1
    mo_core = mf.mo_coeff[:,:ncore]
    mo_cas = C_loc
    hcore = mf.get_hcore()
    energy_nuc = mf.energy_nuc()
    ecore = energy_nuc
    core_dm = np.dot(mo_core, mo_core.conj().T) * 2
    corevhf = mf.get_veff(cell, core_dm)
    # corevhf = lib.chkfile.load('hbn.chk', 'corevhf')
    ecore += np.einsum('ij,ji', core_dm, hcore).real
    ecore += np.einsum('ij,ji', core_dm, corevhf).real * .5
    h1 = reduce(np.dot, (mo_cas.conj().T, hcore+corevhf, mo_cas))
    
    print('H(h1,h2) h1', h1)
    print('diagonalized H(h1, h2) h1', np.linalg.eig(h1))
    
    h2 = my_screened_eris
    
    cell = gto.M() #need a fresh mol object so fci doesn't get huge system's mol info and crash due to lack of memory
    # "incore_anyway=True ensures the customized Hamiltonian (the _eri attribute)
    # is used.  Without this parameter, the MO integral transformation used in
    # subsequent post-HF calculations may
    # ignore the customized Hamiltonian if there is not enough memory."
    cell.incore_anyway = True
    cell.nelectron = ne_act
    
    cisolver = fci.direct_spin1.FCI()
    cisolver.nroots = 4
    
    e, fcivec = cisolver.kernel(h1, h2, ncas, ne_act, ecore=ecore)
    print(e)
    for i, c in enumerate(fcivec):
        print('state = %d, E = %.9f, S^2=%.4f' %
              (i, e[i], fci.spin_op.spin_square(c, ncas, ne_act)[0]))
