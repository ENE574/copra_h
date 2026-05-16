'''
Created on Apr 19, 2017

@author: lradusky
'''
from _collections import defaultdict
import json

import foldxPM_Globals
from  foldxPM_Loader import *

class JSonMolecule( object ):

    def __init__(self, molname=""):
        self.molname = molname
        self.molparams = [ ]
        
        for k in list(foldxPM_Globals.TYPENAMES.keys()):
            self.molparams += [defaultdict(lambda:defaultdict(str))]
            self.molparams[-1]["datastored"] = k
            self.molparams[-1]["datatype"] = foldxPM_Globals.TYPENAMES[k]
            self.molparams[-1][k] = []
        
    def toJson(self, toFile=""):
        retdict = { "molName": self.molname, "molCode": self.threeLetterCode, "molParams": self.molparams}
        
        if toFile == "":
            return json.dumps(retdict, indent="\t")
        else:
            #return foldxPM_Globals.yasFileHandler.writeLine(toFile,json.dumps(retdict, indent="\t"))
            return foldxPM_Globals.yasFileHandler.writeLine(toFile,json.dumps(retdict))
    
    @staticmethod
    def fromJson(filePath):
        loaded_dict = json.loads("\n".join(foldxPM_Globals.yasFileHandler.getLines(filePath)))
        ret = JSonMolecule(loaded_dict["molName"])
        ret.molparams = loaded_dict["molParams"]
        ret.threeLetterCode = loaded_dict["molCode"]
        
        return ret
    
    def setMolName(self,name):
        self.molname = name
    
    def setThreeLetterCode(self,tlc):
        self.threeLetterCode=tlc
    
    def insertIntoTable(self,tableName,record):
        for table in self.molparams:
            if tableName in list(table.keys()):
                # These are uniq
                if tableName in ['AAproperties','scentropy']:
                    table[tableName] = [record]
                else:
                    table[tableName] += [record]
    
    def getAtomsInTable(self, table):
        retList = []
        
        for tableName in self.molparams:
            if table in list(tableName.keys()):
                for subtable in tableName[table]:
                    retList += [subtable[JSonParameterLoader.getAtomFieldName(table)]]
        
        return retList
    
if __name__ == "__main__":
    
    #a = JSonMolecule("Uracil")
    a = JSonMolecule.fromJson("/home/lradusky/Desktop/newMol.json")
    print(a.toJson())
    
    
    
    