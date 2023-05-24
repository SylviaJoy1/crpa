from pyscf.pbc import df
import numpy as np
from pyscf import lib
einsum = lib.einsum
from pyscf.pbc.tools.pbc import madelung
from crpa import cRPA

def kernel(crpa, M=0, screened = True):
    nmo = crpa.nmo
    nocc = crpa.nocc
    nvir = nmo - nocc
    e_mo_occ = crpa.mo_energy[:nocc] 
    e_mo_vir = crpa.mo_energy[nocc:] 
    
    canon_Lov, loc_Lpq = crpa.get_Lpq()
    nact = np.shape(loc_Lpq)[-1]
    
    if not screened:
        unscr_U = einsum('Pij,Pkl->ijkl', loc_Lpq, loc_Lpq)
        for i in range(nact):
            for j in range(i):
                unscr_U[i,i,j,j] += M
                unscr_U[j,j,i,i] += M
            unscr_U[i,i,i,i] += M
        return unscr_U
        
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
    scr_M = M*einsum('Pii,PQ,Qjj->ij', loc_Lpq, i_tilde, loc_Lpq)/einsum('Pii,Pjj->ij', loc_Lpq, loc_Lpq)
    for i in range(nact):
        for j in range(i):
            scr_U[i,i,j,j] += scr_M[i,j]
            scr_U[j,j,i,i] += scr_M[j,i]
        scr_U[i,i,i,i] += scr_M[i,i]
    return scr_U

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
   
#TODO: include ROKS
class cRPA_supercell(cRPA):
    def __init__(self, mf, df_file, loc_coeff=None):
        super().__init__(mf, df_file, loc_coeff=loc_coeff)
        self.M = madelung(mf.cell, kpts=[0,0,0])
    
    def get_Lpq(self):
        canon_Lov, loc_Lpq = get_Lpq(self.mf, self.df_file, self.loc_coeff)
        return canon_Lov, loc_Lpq
    
    def make_U(self):
        U = make_U(self.mf, self.loc_coeff)
        return U
    
    def kernel(self, screened = True):
        self.ERIs = kernel(self, self.M, screened)
        return self.ERIs

if __name__ == '__main__':
    
    #hBN+C2
    from pyscf.pbc import gto, dft
    from crpa import cRPA_CASCI
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
#                mf.with_df._cderi_to_save = 'hbn_c2_dzv_{}.h5'.format(nx)
                mf.with_df._cderi = 'hbn_c2_dzv_{}.h5'.format(nx)
                # mf.chkfile = 'hbn_c2_dzv_{}.chk'.format(nx)
                dm = mf.from_chk('hbn_c2_dzv_{}.chk'.format(nx))
#                mf.chkfile = 'hbn_c2_dzv_{}.chk'.format(nx)
                 
                mf.kernel(dm)
#                mf.kernel()
                
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
#                from pyscf import lo
#                idcs = np.ix_(np.arange(nmo), [nocc-1, nocc])
#                mo_init = lo.PM(cell, mf.mo_coeff[idcs])
#                C_loc = mo_init.kernel()
#                lib.chkfile.dump('hbn_c2_dzv_{}.chk'.format(nx), 'C_loc', C_loc)
                
                C_loc = lib.chkfile.load('hbn_c2_dzv_{}.chk'.format(nx), 'C_loc')
                
                mycRPA = cRPA_supercell(mf, 'hbn_c2_dzv_{}.h5'.format(nx), loc_coeff = C_loc)
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
        for result in pool.map(task, range(2,3)):
            # report the value to show progress
            print(result)
