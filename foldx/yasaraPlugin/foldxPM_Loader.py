'''
Created on May 3, 2017
@author: lradusky
'''

# @TODO: configure plugin with the parameters location

import yasara
#import json
import json
import foldxPM_Globals

PARAMETERS_PATH = [line.split("=")[1].strip() for line in foldxPM_Globals.yasFileHandler.getLines("foldx.cnf")\
                                                if line.split("=")[0].strip() == "FOLDX_ROTABASE"][0]

class JSonParameterLoader( object ):
    
    @staticmethod
    def loadTable(table):
        lines = foldxPM_Globals.yasFileHandler.getLines(PARAMETERS_PATH)
        read = False
        
        json_lines=[]
        for line in lines:
            if line == "JSonDataEnd %s" % table:
                read = False
                
            if read:
                json_lines.append(line)
            
            if line == "JSonDataStart %s" % table:
                read = True

        loaded_dict = json.loads("\n".join(json_lines))
        
        return loaded_dict
        
    @staticmethod
    def getAAFieldName(table):
        if table in ["xyz_water","xyz_sal", "hbondinfo", "H_Pos", "solvenergy", "AAproperties"]:
            return "aa"
    
    @staticmethod
    def getAtomFieldName(table):
        if table in ["xyz_water","xyz_sal","H_Pos"]:
            return "group"
        elif table in ["hbondinfo", "solvenergy"]:
            return "atom"
    
    @staticmethod
    def getMoleculesInTable(table):
        loaded_dict = JSonParameterLoader.loadTable(table)
        
        molecules = set()
        for record in loaded_dict[table]:
            molecules.add(record[JSonParameterLoader.getAAFieldName(table)])
        
        return sorted(list(molecules))
    
    @staticmethod
    def getAtomsInTable(table,molecule):
        loaded_dict = JSonParameterLoader.loadTable(table)

        atoms = set()
        for record in loaded_dict[table]:
            if record[JSonParameterLoader.getAAFieldName(table)] == molecule:
                atoms.add(record[JSonParameterLoader.getAtomFieldName(table)])
        
        return sorted(list(atoms))
    
    @staticmethod
    def getAtomsWithSolvation():
        loaded_dict = JSonParameterLoader.loadTable(table)
        
        atoms = []
        for record in loaded_dict['molParams']:
            if record["datastored"] == "solvenergy":
                for subrecord in record["solvenergy"]:
                    atoms += [subrecord["atom"]]
                    
        return atoms
    
    @staticmethod
    def getAllAtoms():
        names = []
        for i in range(yasara.CountAtom("All")):
            name = yasara.NameAtom(i+1)[0]
            names += [name]

        return names
            
    @staticmethod
    def getRecordsInTable(table,molecule, atom):
        loaded_dict = JSonParameterLoader.loadTable(table)
        
        #print ""
        records = []
        for record in loaded_dict[table]:
            #if record[JSonParameterLoader.getAAFieldName(table)] == "T" and table == "solvenergy":
            #    print record 
            if record[JSonParameterLoader.getAAFieldName(table)] == molecule and record[JSonParameterLoader.getAtomFieldName(table)] == atom:
                records.append(record)
        
        #print ""
        return records
    
    @staticmethod
    def getAApropertiesRecord(molecule):
        loaded_dict = JSonParameterLoader.loadTable("AAproperties")
        
        for record in loaded_dict["AAproperties"]:
            if record["aa"] == molecule:
                return record
    
    @staticmethod
    def getScentropyRecord(molecule):
        loaded_dict = JSonParameterLoader.loadTable("scentropy")

        for record in loaded_dict["scentropy"]:
            if record["aa"] == molecule:
                return record
    
    @staticmethod
    def loadAllFromParametrizedMolecule(newMol):
        molecules = JSonParameterLoader.getMoleculesInTable("AAproperties")
        
        # Select the molecule to copy the parameters
        mol_and_buttons = molecules + ["Button",120,350,"Cancel","Button",280,350,"Continue"]
        win_result = yasara.ShowWin("Custom","Copy from existing... ",400,400,
                      "List",20,50, "Choose a molecule", 300,250, "No", len(molecules),
                      *mol_and_buttons)
        
        print(win_result[0])
        
        if win_result[0]== "Continue":
            sel_mol = win_result[2]
            # For each atom of our molecule
            for sel_atom in JSonParameterLoader.getAllAtoms():
                #print sel_atom
                # For each table
                for table in ["xyz_water","xyz_sal", "hbondinfo", "H_Pos", "hbondinfo", "solvenergy"]:
                    # Get the records of the table for the selected molecule and the target atom
                    records = JSonParameterLoader.getRecordsInTable(table,sel_mol,sel_atom)
                    #if table == 'solvenergy':
                    #    print len(records), [record['atom'] for record in records]
                    
                    for record in records:
                        # Re-define the molecule name in the record
                        record[JSonParameterLoader.getAAFieldName(table)] = newMol.threeLetterCode
                        # And insert this record in the corresponding table
                        newMol.insertIntoTable(table, record)
                
            # For the general tables we have to do the same
            record = JSonParameterLoader.getAApropertiesRecord(sel_mol)
            record["aa"] = newMol.threeLetterCode
            newMol.insertIntoTable("AAproperties", record)
            record = JSonParameterLoader.getScentropyRecord(sel_mol)
            record["aa"] = newMol.threeLetterCode
            newMol.insertIntoTable("scentropy", record)
            
            newMol.toJson(toFile='tmp/tmp.json')
    
    @staticmethod
    def loadRecordFromTable(newMol, atom, table, description):
        
        molecules = JSonParameterLoader.getMoleculesInTable(table)
        # Lets load the molecules which have water rotamers
        mol_and_buttons = molecules + ["Button",120,350,"Cancel","Button",280,350,"Continue"]
        win_result = yasara.ShowWin("Custom","%s from existing..." % description,400,400,
                      "List",20,50, "Choose the molecule to copy %s:" % description,300,250, "No", len(molecules),
                      *mol_and_buttons)
        
        if win_result[0]== "Continue":
            sel_mol = win_result[2]
            atomList = JSonParameterLoader.getAtomsInTable(table,sel_mol)
            # Lets atoms of the selected molecule which have water rotamers
            atom_and_buttons = atomList + ["Button",120,350,"Cancel","Button",280,350,"Continue"]
            win_result = yasara.ShowWin("Custom","%s from existing..." % description,400,400,
                      "List",20,50, "Choose the atom to copy %s:" % description,300,250, "No", len(atomList),
                      *atom_and_buttons)
        
        if win_result[0]== "Continue":
            sel_atom = win_result[2]
            records = JSonParameterLoader.getRecordsInTable(table,sel_mol,sel_atom)
            for record in records:
                record[JSonParameterLoader.getAAFieldName(table)] = newMol.threeLetterCode
                record[JSonParameterLoader.getAtomFieldName(table)] = atom.name
                
                newMol.insertIntoTable(table, record)
                
            newMol.toJson(toFile='tmp/tmp.json')
            
    @staticmethod
    def GetInputTexts(header, labels, cancelCaption, acceptCaption):
        params = []
        for i in range(len(labels)):
            params += ["TextInput",20,50+70*i,labels[i],300,10]
            
        params += ["Button",280,350,acceptCaption]
        params += ["Button",120,350,cancelCaption]
        
        return yasara.ShowWin("Custom",header,400,400,*params) 

    @staticmethod
    def GetInputChoice(header, labels, cancelCaption, acceptCaption):
        
        params = []
        for i in range(len(labels)):
            params += [30,60+50*i,labels[i]]
            
        params += ["Button",280,350,acceptCaption]
        params += ["Button",120,350,cancelCaption]
        
        return yasara.ShowWin("Custom",header,400,400,"RadioButtons",len(labels),1, *params)
         
    
    @staticmethod
    def GetInputList(header, description, labels, acceptCaption, multiple="No"):
        len_list=len(labels)
        params = labels
        params += ["Button",280,350,acceptCaption]
        
        return yasara.ShowWin("Custom",header,400,400, "List",20,50, description ,300,250, multiple, len_list, *params)
        
    
    
    