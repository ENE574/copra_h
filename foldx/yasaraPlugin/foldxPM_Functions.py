'''
Created on May 3, 2017
@author: lradusky
'''

from foldxPM_Globals import *
from foldxPM_Mol import *

def new_aaprop(molecule, natural, molweight, extinction, DGmax, movingProtons, \
               virtualProtonsHolder, numberOfMovingProtons, isAlcohol, protonationStates, misplacedDihed):
    return \
        dict(list(zip(FIELDS[TYPENAMES["AAproperties"]],[\
               molecule.threeLetterCode, int(natural), float(molweight), float(extinction), float(DGmax), int(movingProtons), \
               int(virtualProtonsHolder), int(numberOfMovingProtons), int(isAlcohol), int(protonationStates), int(misplacedDihed)\
                                                    ])))
        
def new_scentropy(molecule, entropy, radius, centre_atom, second_atom):
    return dict(list(zip(FIELDS[TYPENAMES["scentropy"]],[molecule.threeLetterCode, 0.0, float(entropy), radius, centre_atom, second_atom])))
        
def new_water_rotamer(molecule, atomName, x, y, z):
    return dict(list(zip(FIELDS["water_rotamer"],[\
            "HOH", "O", molecule.threeLetterCode, atomName, float(x), float(y), float(z), -1\
        ])))
    
    
def new_hbondinfo(molecule, atomName, donor, acceptor, number_of_H, number_dummy_H, 
                  moving_H, double_bond, charge, Hname, pKa, dipoled, charged, min_len_hbond,
                  Hbond_plane_dist, partcov, explicit_solv, tolerance_hbond, hybridization):
    
    return dict(list(zip(FIELDS["Hbonds"],[\
            molecule.threeLetterCode, atomName, int(donor), int(acceptor), int(number_of_H), int(number_dummy_H), 
                  int(moving_H), int(double_bond), float(charge), Hname, float(pKa), int(dipoled), int(charged), float(min_len_hbond),
                  float(Hbond_plane_dist), float(partcov), float(explicit_solv), float(tolerance_hbond), int(hybridization)
        ])))

def new_H_Pos(molecule, atomName, partner1, pospart1, partner2, pospart2, hydrogen, x, y, z, isvirtual, isexplicit, protonated, is_carbonyl):
    
    return dict(list(zip(FIELDS["position_hydrogen"],[\
            molecule.threeLetterCode, atomName, partner1, int(pospart1), partner2, int(pospart2), hydrogen, \
            float(x), float(y), float(z), int(isvirtual), int(isexplicit), int(protonated), int(is_carbonyl)\
        ])))

def new_solvenergy(molecule, atomName, atom_type, res_type, \
                   volume, Occ, Occmax, vdw, enerG, level, min_radius, radius, \
                   hydrophobic, backbone_atom, polar, cycle, xTemplate, yTemplate, zTemplate,\
                   omega, phi, psi, chi1, chi2, chi3, chi4, chi5, chi6, chi7, lastDihedral, molecule_type, neigh_atom):
        
    # The three letter code has to be deleted
    return dict(list(zip(FIELDS["tablesisolvation"][0:-1]+["neigh_atom"],[\
            molecule.threeLetterCode, " ", atom_type, res_type, atomName, \
            float(volume), float(Occ), float(Occmax), float(vdw), float(enerG), int(level), float(min_radius), float(radius), \
            int(hydrophobic), int(backbone_atom), int(polar), int(cycle), float(xTemplate), float(yTemplate), float(zTemplate),\
            int(omega), int(phi), int(psi), int(chi1), int(chi2), int(chi3), int(chi4), int(chi5), int(chi6), int(chi7), \
            int(lastDihedral), int(molecule_type)] + [neigh_atom])))



