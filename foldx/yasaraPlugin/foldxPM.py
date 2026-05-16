# YASARA PLUGIN
# TOPIC:       Parametrize new molecules for FoldX
# TITLE:       Parametrize new molecules for FoldX
# AUTHOR:      Leandro Radusky
# LICENSE:     GPL
# DESCRIPTION: Parametrize new molecules for FoldX
#
# This is a YASARA plugin to be placed in the yasara/plg subdirectory
# Go to www.yasara.org/repository for documentation and downloads

"""
MainMenu: Analyze
  PullDownMenu: Foldx Molecule Handling
    PopUpMenu: Load molecule parameters
      TextInputMenu_MolName: General properties of the molecule
        Text: General Properties
        Text: Type the name of the molecule
      TextInputMenu_ThreeLetterCode: General properties of the molecule
        Text: General Properties
        Text: Type the three letter code of this molecule
      Request: LoadMoleculeParameters
    PopUpMenu: Parametrize Molecule
      TextInputMenu_MolName: General properties of the molecule
        Text: General Properties
        Text: Type the name of the molecule
      TextInputMenu_ThreeLetterCode: General properties of the molecule
        Text: General Properties
        Text: Type the three letter code of this molecule
      Request: StartParametrization

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Define General Parameters 
    Request: AddGeneralParameters

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Load general parameters from existing molecule 
    Request: LoadGeneralParameters

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Add Water Rotamer 
    Request: Add-xyz_water
    
AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Add Water Rotamer from existing molecule
    Request: AddExisting-xyz_water

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Add Salt Rotamer
    Request: Add-xyz_sal
    
AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Add Salt Rotamer from existing molecule
    Request: AddExisting-xyz_sal

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Add Hydrogen Bond info 
    Request: Add-hbondinfo

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Add Hydrogen Bond info from existing molecule
    Request: AddExisting-hbondinfo

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Add Hydrogen Position 
    Request: Add-H_Pos

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Add Hydrogen Position from existing molecule
    Request: AddExisting-H_Pos

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Add Solvation parameters 
    Request: Add-solvenergy

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Add Solvation parameters from existing molecule
    Request: AddExisting-solvenergy

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Remark Atoms with defined water rotamers
    Request: ColorAtoms-xyz_water

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Remark Atoms with defined salt rotamers
    Request: ColorAtoms-xyz_sal

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Remark Atoms with defined H-Bond information
    Request: ColorAtoms-hbondinfo

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Remark Atoms with defined H-Position information
    Request: ColorAtoms-H_Pos

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Remark Atoms with defined Solvation information
    Request: ColorAtoms-solvenergy

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Save Molecule Parameters to file
    Request: SaveMolecule

AtomContextMenu after Swap: FoldX Molecule Handling
  PopUpMenu: Load Molecule Parameters from file
    Request: LoadMolecule

"""

#!/usr/bin/python
import yasara  # @UnresolvedImport
from python2to3 import *
from foldxPM_Mol import *
from foldxPM_Globals import *
from foldxPM_Functions import *
from foldxPM_Loader import *
# -----------------------------------------------------------
#  LOAD ALL THE "LOADABLE" PARAMETERS FROM OTHER MOLECULE 
# -----------------------------------------------------------
if (yasara.request == "LoadMoleculeParameters"):
    try:
        yasFileHandler.removeFile('tmp/tmp.json')
    except:
        pass
    
    newMol = JSonMolecule(yasara.selection[0].text[0])
    newMol.setThreeLetterCode(yasara.selection[1].text[0])
    
    JSonParameterLoader.loadAllFromParametrizedMolecule(newMol)
    
    newMol.toJson(toFile='tmp/tmp.json')

#!/usr/bin/python
import yasara  # @UnresolvedImport
from python2to3 import *
from foldxPM_Mol import *
from foldxPM_Globals import *
from foldxPM_Functions import *
from foldxPM_Loader import *
# ---------------------------------------------
#  START A MOLECULE PARAMETRIZATION   
# ---------------------------------------------
if (yasara.request=="StartParametrization"):
    try:
        yasFileHandler.removeFile('tmp/tmp.json')
    except:
        pass
    
    newMol = JSonMolecule(yasara.selection[0].text[0])
    newMol.setThreeLetterCode(yasara.selection[1].text[0])
    
    yasara.ShowMessage("Now, right clicking on the molecule atoms you can fill all the parameters needed.")
    
    newMol.toJson(toFile='tmp/tmp.json')

import yasara  # @UnresolvedImport
from python2to3 import *
from foldxPM_Mol import *
from foldxPM_Globals import *
from foldxPM_Functions import *
from foldxPM_Loader import *
# ---------------------------------------------
#  DEFINE GENERAL PARAMETERS OF THE MOLECULE   
# ---------------------------------------------
if (yasara.request=="AddGeneralParameters"):
    newMol = JSonMolecule.fromJson('tmp/tmp.json')
    header = "General Parameters"
    cancelCaption= "Cancel"
    acceptCaption= "Continue"
    
    labels = ["This is a natural molecule", "This is NOT a natural molecule"]
        
    (action,natural) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
    natural = int(natural) % 2
    
    if action == "Continue":
        labels = ["Molecular weight:", "Extinction coefficient:","Max DeltaG: "]

        (action, molweight, extinction, DGmax) = JSonParameterLoader.GetInputTexts(header,labels, cancelCaption, acceptCaption)
    
    if action == "Continue":
        labels = ["Number of moving protons: ", "Entropy: ", "Radius: "]
        
        (action, numberOfMovingProtons, entropy_abagyan, radius) = JSonParameterLoader.GetInputTexts(header,labels, cancelCaption, acceptCaption)
        movingProtons = 0 if int(numberOfMovingProtons) <= 0 else 1 
        if movingProtons == 0:
            virtualProtonsHolder = -1
            movingProtons = -1
        else:
            #show numbered atoms from solvation table
            header = "Pick Virtual Protons holder"
            description = "Protons holder"
            labels = JSonParameterLoader.getAtomsWithSolvation()
            cancelCaption = "Cancel"
            acceptCaption = "Continue"
            
            if len(labels) > 0:
                win_result = JSonParameterLoader.GetInputList(header, description, labels, acceptCaption)
                action = "Continue" if win_result[0] == 1 else "Cancel"
                virtualProtonsHolder = labels.index(win_result[1])
            else:
                #if there are not parametrized, show a warning
                yasara.ShowMessage("To define virtual protons, you have to define the solvation parameters of the atoms first")
                action = "Cancel"
                
    if action == "Continue":
        labels = ["This molecule is an alcohol", "This molecule is NOT an alcohol"]
        
        (action,isAlcohol) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        isAlcohol = int(isAlcohol) % 2

    if action == "Continue":
        labels = ["There are protonation states", "There are NOT protonation states"]
        
        (action,protonationStates) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        protonationStates = int(protonationStates) % 2
    
    if action == "Continue":
        labels = ["There are misplaced dihedrals", "There are NOT misplaced dihedrals"]
        
        (action,misplacedDihed) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        misplacedDihed = int(misplacedDihed) % 2
    
    if action == "Continue":
        #show numbered atoms from solvation table
        header = "Pick Center Atom"
        description = "Center Atom"
        labels = JSonParameterLoader.getAtomsWithSolvation()
        cancelCaption = "Cancel"
        acceptCaption = "Continue"
        
        if len(labels) > 0:
            win_result = JSonParameterLoader.GetInputList(header, description, labels, acceptCaption)
            action = "Continue" if win_result[0] == 1 else "Cancel"
            center_atom = labels.index(win_result[1])
        else:
            #if there are not parametrized, show a warning
            yasara.ShowMessage("To define the center atom, you have to define the solvation parameters of the atoms first")
            action = "Cancel"
    
    if action == "Continue":
        #show numbered atoms from solvation table
        header = "Pick Second Atom"
        description = "Second Atom"
        labels = JSonParameterLoader.getAtomsWithSolvation()
        cancelCaption = "Cancel"
        acceptCaption = "Continue"
        
        if len(labels) > 0:
            win_result = JSonParameterLoader.GetInputList(header, description, labels, acceptCaption)
            action = "Continue" if win_result[0] == 1 else "Cancel"
            second_atom = labels.index(win_result[1])
        else:
            #if there are not parametrized, show a warning
            yasara.ShowMessage("To define the center atom, you have to define the solvation parameters of the atoms first")
            action = "Cancel"
    
    if action == "Continue":
        newMol.insertIntoTable("AAproperties", new_aaprop(newMol, natural, molweight, extinction, DGmax, movingProtons, \
               virtualProtonsHolder, numberOfMovingProtons, isAlcohol, protonationStates, misplacedDihed))
        newMol.insertIntoTable("scentropy", new_scentropy(newMol, entropy_abagyan, radius, center_atom, second_atom))
        
        newMol.toJson(toFile='tmp/tmp.json')

import yasara  # @UnresolvedImport
from python2to3 import *
from foldxPM_Mol import *
from foldxPM_Globals import *
from foldxPM_Functions import *
from foldxPM_Loader import *
# ---------------------------------------------
#      ADD WATER ROTAMERS
# ---------------------------------------------
if (yasara.request=="Add-xyz_water") or \
    (yasara.request=="Add-xyz_sal"):
    newMol = JSonMolecule.fromJson('tmp/tmp.json')
    atom = yasara.selection[0].atom[0]
    
    header = "Let's define the water coordinates linked to this atom"
    labels = ["X coordinate", "Y coordinate", "Z coordinate"]
    cancelCaption = "Cancel"
    acceptCaption = "Save"

    (action, x, y, z) = JSonParameterLoader.GetInputTexts(header,labels, cancelCaption, acceptCaption)
    
    if action == "Save":
        if yasara.request == "Add-xyz_water":
            newMol.insertIntoTable("xyz_water", new_water_rotamer(newMol,atom.name,x,y,z))
        else:
            newMol.insertIntoTable("xyz_sal", new_water_rotamer(newMol,atom.name,x,y,z))
    
        newMol.toJson(toFile='tmp/tmp.json')

import yasara  # @UnresolvedImport
from python2to3 import *
from foldxPM_Mol import *
from foldxPM_Globals import *
from foldxPM_Functions import *
from foldxPM_Loader import *
# ---------------------------------------------
#      ADD H-BOND
# ---------------------------------------------
if (yasara.request=="Add-hbondinfo"):
    newMol = JSonMolecule.fromJson('tmp/tmp.json')
    atom = yasara.selection[0].atom[0]
    
    header = "Let's define the HBond properties for this atom"
    labels = ["Number of donor possibilities", "Number of aceptor possibilities", "Number of possible H or waters", "Number of H placed by the subroutine"]
    cancelCaption = "Cancel"
    acceptCaption = "Continue"
    
    (action, donor, acceptor, number_of_H, number_dummy_H) = JSonParameterLoader.GetInputTexts(header,labels, cancelCaption, acceptCaption)
    
    if action == "Continue":
        labels = ["Charge of the atom", "Atom on which the donor atom is bound", "pKa of the atom", "Minimal distance for Hbonding"]
    
        (action, charge, Hname, pKa, min_len_hbond) = JSonParameterLoader.GetInputTexts(header,labels, cancelCaption, acceptCaption)
        
    if action == "Continue":
        labels = ["Atom allow double bonds", "Atom DONT allow double bonds"]
        
        (action,double_bond) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        double_bond = int(double_bond) % 2

    if action == "Continue":
        labels = ["Plane distance", "Default partial covalent contribution","Explicit solvent value","HBond tolerance"]

        (action, Hbond_plane_dist, partcov, explicit_solv, tolerance_hbond) = JSonParameterLoader.GetInputTexts(header,labels, cancelCaption, acceptCaption)

    if action == "Continue":
        
        labels = ["SP2_N_ORB1", "SP2_N_H1", "SP2_N_H2", "SP2_O_ORB2", "SP3_O_H1ORB2", "SP3_N_H3", "NO_HYBRID"]
        description = "Choose the hybridization:"
        
        win_result = JSonParameterLoader.GetInputList(header, description, labels, acceptCaption)
        action = "Continue" if win_result[0] == 1 else "Cancel"
        hybridization = [i for i, j in enumerate(labels) if j == win_result[1]][0]
        
    if action == "Continue":
        labels = ["The H can rotate", "The H is restricted"]
        
        (action,moving_H) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        moving_H = int(moving_H) % 2
        
    if action == "Continue":
        labels = ["This atom is dipoled", "This atom is NOT dipoled"]
        
        (action,dipoled) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        dipoled = int(dipoled) % 2
    
    if action == "Continue":
        labels = ["This atom is charged", "This atom is NOT charged"]
        
        (action,charged) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        charged = int(charged) % 2
    
    if action == "Continue":
        newMol.insertIntoTable("hbondinfo", new_hbondinfo(newMol,atom.name, donor, acceptor, number_of_H, number_dummy_H,\
                            moving_H, double_bond, charge, Hname, pKa, dipoled, charged, min_len_hbond,\
                            Hbond_plane_dist, partcov, explicit_solv, tolerance_hbond, hybridization))
    
        newMol.toJson(toFile='tmp/tmp.json')

import yasara  # @UnresolvedImport
from python2to3 import *
from foldxPM_Mol import *
from foldxPM_Globals import *
from foldxPM_Functions import *
from foldxPM_Loader import *
# ---------------------------------------------
#      ADD HYDROGEN POSITIONS
# ---------------------------------------------
if (yasara.request=="Add-H_Pos"):
    newMol = JSonMolecule.fromJson('tmp/tmp.json')
    atom = yasara.selection[0].atom[0]
    
    labels = ["SP2_N_ORB1", "SP2_N_H1", "SP2_N_H2", "SP2_O_ORB2", "SP3_O_H1ORB2", "SP3_N_H3", "NO_HYBRID"]
    description = "Choose the hybridization:"
    
    
    header = "Choose the Partner 1 of this atom."
    labels = JSonParameterLoader.getAllAtoms()
    cancelCaption = "Cancel"
    acceptCaption = "Continue"

    win_result = JSonParameterLoader.GetInputList(header, description, labels, acceptCaption)
    action = "Continue" if win_result[0] == 1 else "Cancel"
    partner1 = win_result[1]
    
    if action == "Continue":
        header = "Choose the Partner 2 of this atom."
        
        win_result = JSonParameterLoader.GetInputList(header, description, labels, acceptCaption)
        action = "Continue" if win_result[0] == 1 else "Cancel"
        partner2 = win_result[1]
    
    if action == "Continue":
        header = "Hydrogen Position Properties"
        labels = ["Name of the H atom", "X coordinate", "Y coordinate", "Z coordinate"]
        
        (action, hydrogen, x, y, z) = JSonParameterLoader.GetInputTexts(header,labels, cancelCaption, acceptCaption)

    if action == "Continue":
        labels = ["This hydrogen is virtual", "This hydrogen is NOT virtual"]
        
        (action,isvirtual) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        isvirtual = int(isvirtual) % 2
        
    if action == "Continue":
        labels = ["This hydrogen is explicit", "This hydrogen is NOT explicit"]
        
        (action,isexplicit) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        isexplicit = int(isexplicit) % 2
    
    if action == "Continue":
        labels = ["This atom is protonated", "This atom is NOT protonated"]
        
        (action,protonated) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        protonated = int(protonated) % 2
        
    if action == "Continue":
        labels = ["This atom is carbonyl", "This atom is NOT carbonyl"]
        
        (action,is_carbonyl) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        is_carbonyl = int(is_carbonyl) % 2
        
    
    if action == "Continue":
        # This values are always 0, its Ok?
        pospart1 = 0
        pospart2 = 0
        newMol.insertIntoTable("H_Pos", new_H_Pos(newMol,atom.name,\
                                                  partner1, pospart1, partner2, pospart2, hydrogen, \
                                                  x, y, z, isvirtual, isexplicit, protonated, is_carbonyl))
        
        newMol.toJson(toFile='tmp/tmp.json')

import yasara  # @UnresolvedImport
from python2to3 import *
from foldxPM_Mol import *
from foldxPM_Globals import *
from foldxPM_Functions import *
from foldxPM_Loader import *
# ---------------------------------------------
#      ADD SOLVATION PARAMETERS
# ---------------------------------------------
if (yasara.request=="Add-solvenergy"):
    newMol = JSonMolecule.fromJson('tmp/tmp.json')
    atom = yasara.selection[0].atom[0]
    
    cancelCaption = "Cancel"
    acceptCaption = "Continue"
    
    header = "Atom Type"
    labels = ["This is an ATOM", "This is an HETATM"]
    (action,atom_type) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
    atom_type = "ATOM" if int(atom_type) == 1 else "HETATM"
    
    if action == "Continue":
        header = "Residue Type"
        labels = ["This molecule is PROTEIN", \
                  "This molecule is DNA", \
                  "This molecule is SMALLMOLECULE", \
                  "This molecule is SINGLEATOMLIGAND", \
                  ]
        
        (action,res_type) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        if int(res_type) == 1:
            res_type = "PROTEIN"
        elif int(res_type) == 2:
            res_type = "DNA"
        elif int(res_type) == 3:
            res_type = "SMALLMOLECULE"
        elif int(res_type) == 4:
            res_type = "SINGLEATOMLIGAND"
    
    if action == "Continue":
        header = "Solvation Properties"
        labels = ["Atom Volume", "Minimal volumetric occupancy", "Maximal volumetric occupancy", "VdWaals unscaled energies"]
        
        (action, volume, Occ, Occmax, vdw) = JSonParameterLoader.GetInputTexts(header,labels, cancelCaption, acceptCaption)
    
    if action == "Continue":
        header = "Solvation Properties"
        labels = ["Hydrophobic and Polar solvation E unscaled", "Atom radius used for internal vdWaals", "Atom radius to detect vdWaels clashes"]
        
        (action, enerG, min_radius, radius) = JSonParameterLoader.GetInputTexts(header,labels, cancelCaption, acceptCaption)
    
    if action == "Continue":
        header = "Solvation Properties"
        description = "Select Atom Level"
        labels = ["LEVEL_O", "LEVEL_N", "LEVEL_A", "LEVEL_B", "LEVEL_G", "LEVEL_D", "LEVEL_E", "LEVEL_Z", "LEVEL_H", "LEVEL_I", "LEVEL_K"]
        
        win_result = JSonParameterLoader.GetInputList(header, description, labels, acceptCaption)
        action = "Continue" if win_result[0] == 1 else "Cancel"
        level = labels.index(win_result[1])-1
    
    if action == "Continue":
        labels = ["This atom is hydrophobic", "This atom is NOT hydrophobic"]
        
        (action,hydrophobic) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        hydrophobic = int(hydrophobic) % 2
    
    if action == "Continue":
        labels = ["This atom belongs to backbone", "This atom DON'T belongs to backbone"]
        
        (action,backbone_atom) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        backbone_atom = int(backbone_atom) % 2
    
    if action == "Continue":
        labels = ["This atom is polar", "This atom is NOT polar"]
        
        (action,polar) = JSonParameterLoader.GetInputChoice(header,labels, cancelCaption, acceptCaption)
        polar = int(polar) % 2
    
    if action == "Continue":
        header = "Solvation Properties"
        labels = ["Belongs to cycle Num? (Cero if dont)", "X Template Coord", "Y Template Coord", "Z Template Coord"]
        
        (action, cycle, xTemplate, yTemplate, zTemplate) = JSonParameterLoader.GetInputTexts(header,labels, cancelCaption, acceptCaption)
    
    if action == "Continue":
        header = "Solvation Properties"
        labels = ["Omega angle", "Phi angle", "Psi angle"]
        
        (action, omega, phi, psi) = JSonParameterLoader.GetInputTexts(header,labels, cancelCaption, acceptCaption)
    
    if action == "Continue":
        header = "Solvation Properties"
        labels = ["Chi1 angle", "Chi2 angle", "Chi3 angle", "Chi4 angle"]
        
        (action, chi1, chi2, chi3, chi4) = JSonParameterLoader.GetInputTexts(header,labels, cancelCaption, acceptCaption)

    if action == "Continue":
        header = "Solvation Properties"
        labels = ["Chi5 angle", "Chi6 angle", "Chi7 angle"]
        
        (action, chi5, chi6, chi7) = JSonParameterLoader.GetInputTexts(header,labels, cancelCaption, acceptCaption)
    
    if action == "Continue":
        header = "Solvation Properties"
        description = "Last dihedral defined angle"
        labels = ["OMEGA", "PHI", "PSI", "CHI1", "CHI2", "CHI3", "CHI4", "CHI5", "CHI6", "CHI7", "DUMMY"]
        
        win_result = JSonParameterLoader.GetInputList(header, description, labels, acceptCaption)
        action = "Continue" if win_result[0] == 1 else "Cancel"
        lastDihedral = labels.index(win_result[1]) if win_result[1] != "DUMMY" else "999"
    
    if action == "Continue":
        header = "Solvation Properties"
        description = "Molecule type"
        labels = ["Protein", "Water", "DNA", "Heteroatom"]
        
        win_result = JSonParameterLoader.GetInputList(header, description, labels, acceptCaption)
        action = "Continue" if win_result[0] == 1 else "Cancel"
        molecule_type = labels.index(win_result[1]) 
    
    if action == "Continue":
        header = "Solvation Properties"
        description = "Neighbour Atoms"
        labels = JSonParameterLoader.getAllAtoms()
        
        win_result = JSonParameterLoader.GetInputList(header, description, labels, acceptCaption, "Yes")
        
        action = "Continue" if win_result[0] >= 1 else "Cancel"
        neigh_atom = win_result[1:] + ["999"]*16
        neigh_atom = neigh_atom[0:16]
         
    
    if action == "Continue":
        newMol.insertIntoTable("solvenergy", new_solvenergy(newMol,atom.name,\
                                        atom_type, res_type, \
                                        volume, Occ, Occmax, vdw, \
                                        enerG, level, min_radius, radius, \
                                        hydrophobic, backbone_atom, polar, cycle, xTemplate, yTemplate, zTemplate,\
                                        omega, phi, psi, chi1, chi2, chi3, chi4, chi5, chi6, chi7, \
                                        lastDihedral, molecule_type, neigh_atom))
        
        newMol.toJson(toFile='tmp/tmp.json')


import yasara  # @UnresolvedImport
from python2to3 import *
from foldxPM_Mol import *
from foldxPM_Globals import *
from foldxPM_Functions import *
from foldxPM_Loader import *
# ---------------------------------------------
#    ADD PARAMETERS FROM EXISTING MOLECULE
# ---------------------------------------------
if (yasara.request=="AddExisting-xyz_water") or\
   (yasara.request=="AddExisting-xyz_sal") or\
   (yasara.request=="AddExisting-hbondinfo") or\
   (yasara.request=="AddExisting-H_Pos") or\
   (yasara.request=="AddExisting-solvenergy"):
    
    newMol = JSonMolecule.fromJson('tmp/tmp.json')
    atom = yasara.selection[0].atom[0]
    
    table = yasara.request.split("-")[1]
    description = "..."
    
    JSonParameterLoader.loadRecordFromTable(newMol, atom, table, description)


import yasara  # @UnresolvedImport
from python2to3 import *
from foldxPM_Mol import *
from foldxPM_Globals import *
from foldxPM_Functions import *
from foldxPM_Loader import *
# ---------------------------------------------
#  COLOR ATOMS OF A TABLE
# ---------------------------------------------
if (yasara.request=="ColorAtoms-xyz_water") or\
  (yasara.request=="ColorAtoms-xyz_sal") or\
   (yasara.request=="ColorAtoms-hbondinfo") or\
   (yasara.request=="ColorAtoms-H_Pos") or\
   (yasara.request=="ColorAtoms-solvenergy"):
    newMol = JSonMolecule.fromJson('tmp/tmp.json')
    
    yasara.ColorAtom("All","White")
    
    for atomName in newMol.getAtomsInTable(yasara.request.split("-")[1]):
        print(atomName)
        yasara.ColorAtom(atomName,"Green")

import yasara  # @UnresolvedImport
from python2to3 import *
from foldxPM_Mol import *
from foldxPM_Globals import *
from foldxPM_Functions import *
from foldxPM_Loader import *
# ---------------------------------------------
#  SAVE MOLECULE TO JSON FILE
# ---------------------------------------------
if (yasara.request=="SaveMolecule"):
    newMol = JSonMolecule.fromJson('tmp/tmp.json')
    win_result = yasara.ShowWin("Custom","Let's define the HBond properties for this atom",400,400,
                          "TextInput",20,50,"Define the path of your JSon File",300,150,
                          "Button",280,350,"Save",
                          "Button",120,350,"Cancel")
    
    (action, path) = win_result
    
    if action == "Save":
        newMol.toJson(toFile=path)

import yasara  # @UnresolvedImport
from python2to3 import *
from foldxPM_Mol import *
from foldxPM_Globals import *
from foldxPM_Functions import *
from foldxPM_Loader import *
# ---------------------------------------------
#  LOAD MOLECULE PARAMETERS FROM JSON FILE
# ---------------------------------------------
if (yasara.request=="LoadMolecule"):
    win_result = yasara.ShowWin("Custom","Load parameters from a saved file",400,400,
                          "TextInput",20,50,"Write the path of your JSon File",300,150,
                          "Button",280,350,"Load",
                          "Button",120,350,"Cancel")
    
    (action, path) = win_result
    
    if action == "Load":
        newMol = JSonMolecule.fromJson(path)
        newMol.toJson(toFile='tmp/tmp.json')

# -----------------------------------------
#  ALWAYS FINISH WITH THIS
# -----------------------------------------
yasara.plugin.end()
# -----------------------------------------




