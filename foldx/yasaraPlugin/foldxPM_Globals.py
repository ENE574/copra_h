'''
Created on Apr 19, 2017
@author: lradusky
'''

from _collections import defaultdict

#Types
TYPENAMES = defaultdict(str)
TYPENAMES["xyz_sal"] = "water_rotamer"
TYPENAMES["xyz_water"] = "water_rotamer"
TYPENAMES["AAproperties"] = "aaprops"
TYPENAMES["scentropy"] = "tablesidechainentropy"
TYPENAMES["hbondinfo"] = "Hbonds"
TYPENAMES["H_Pos"] = "position_hydrogen"
TYPENAMES["solvenergy"] = "tablesisolvation"

#Fields
FIELDS = defaultdict(str)
FIELDS["water_rotamer"] = ["ion_atom","ion_aa","aa","group","x","y","z","reference_index"]
FIELDS["aaprops"] = ["aa","natural","molweight","extinction","DGmax","movingProtons","virtualProtonsHolder",
                  "numberOfMovingProtons","isAlcohol","protonationStates","misplacedDihed"]
FIELDS["tablesidechainentropy"]=["aa", "entropy", "entropy_abagyan", "radius", "centre_atom", "second_atom"]
FIELDS["Hbonds"] = ["aa","atom","donor","acceptor","number_of_H","number_dummy_H","moving_H",
                    "double_bond","charge","Hname","pKa","dipoled","charged","min_len_hbond",
                    "Hbond_plane_dist","partcov","explicit_solv","tolerance_hbond","hybridization"]
FIELDS["position_hydrogen"]=["aa", "group", "partner1", "pospart1", "partner2", "pospart2", 
                             "hydrogen", "x", "y", "z", "isvirtual", "isexplicit", "protonated", "is_carbonyl"]
FIELDS["tablesisolvation"]=["aa","onelet","atom_type","res_type","atom","volume","Occ","Occmax","vdw",
                            "enerG","level","min_radius","radius","hydrophobic","backbone_atom","polar",
                            "cycle","xTemplate","yTemplate","zTemplate","omega","phi","psi","chi1","chi2",
                            "chi3","chi4","chi5","chi6","chi7","lastDihedral","molecule_type","neigh_atom___16"]

import os, gzip

class yasFileHandler(object):
    '''
    classdocs
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
    
    @staticmethod
    def fileExists(path):
        try:
            with open(path): return True
        except IOError:
            return False
    
    @staticmethod
    # @param path: the path of the file to be readed
    # @return: a list of string with all the lines, [] if error 
    def getGZLines(path):
        try:
            f = gzip.open(path, 'r')
            lines = f.readlines()
            f.close()
            ret = []
            for l in lines:
                ret += [l.strip()+"\n"]
            return ret
        except Exception as inst:
            return []
    
    @staticmethod
    # @param path: the path of the file to be readed
    # @return: a list of string with all the lines, [] if error 
    def getLines(path):
        try:
            f = open(path, 'r')
            lines = f.readlines()
            f.close()
            ret = []
            for l in lines:
                ret += [l.strip()]
            return ret
        except Exception as inst:
            return []
    
    @staticmethod
    # @param path: the path of the file to be readed
    # @return: a string with the file 
    def getString(path):
        try:
            return "\n".join(yasFileHandler.getLines(path))
        except Exception as inst:
            return ""

    @staticmethod
    # @param path: the path of the file where the line will be appended
    # @param line: a string to append in a new line 
    # @return: True is success, False if don't
    def appendLine(path, line):
        try:
            yasFileHandler.ensureDir(path)
            f = open(path, 'a')
            f.write(line+"\n")
            f.close()
            return True
        except Exception as inst:
            return False
    
    @staticmethod
    # @param path: the path of the file where the line will be appended
    # @param lines: a list of strings to append in a new line 
    # @return: True is success, False if don't
    def appendLines(path, lines):
        try:
            yasFileHandler.ensureDir(path)
            f = open(path, 'a')
            for line in lines:
                f.write(line+"\n")
            f.close()
            return True
        except Exception as inst:
            return False
    
    @staticmethod
    # @param path: the path of the file where the line will be written
    # @param line: a string to write in a new line 
    # @return: True is success, False if don't
    def writeLine(path, line, addReturn = "\n"):
        try:
            yasFileHandler.ensureDir(path)
            f = open(path, 'w')
            f.write(line+addReturn)
            f.close()
            return True
        except Exception as inst:
            return False
    
    @staticmethod
    # @param path: the path of the file where the line will be written
    # @param line: a string to write in a new line 
    # @return: True is success, False if don't
    def writeLines(path, lines):
        try:
            yasFileHandler.ensureDir(path)
            f = open(path, 'w')
            for line in lines:
                if len(line)== 0 or line[-1] == "\n":
                    f.write(line)
                else:
                    f.write(line+"\n")
            f.close()
            return True
        except Exception as inst:
            return False
    
    @staticmethod
    # @param path: the dir to make if not exist
    # @summary: make dir if not exists 
    def ensureDir(path):
        d = os.path.dirname(path)
        if not os.path.exists(d):
            os.makedirs(d)
        return d
    
    @staticmethod
    # @param path: the dir to make if not exist
    # @summary: make dir if not exists 
    def exists(path):
        return os.path.exists(path)
    
    @staticmethod
    def removeFile(path):
        return os.remove(path)