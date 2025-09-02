# 20种氨基酸常量
ALA = 0
CYS = 1
ASP = 2
GLU = 3
PHE = 4
GLY = 5
HIS = 6
ILE = 7
LYS = 8
LEU = 9
MET = 10
ASN = 11
PRO = 12
GLN = 13
ARG = 14
SER = 15
THR = 16
VAL = 17
TRP = 18
TYR = 19

# 共价半径 (Å)
covalent_radius = {
    "C": 0.77,
    "N": 0.70,
    "O": 0.66,
    "S": 1.05,
    "H": 0.37,
    "P": 1.10,
    "F": 0.64,
    "Cl": 1.00,
    "Br": 1.14,
    "I": 1.33,
    "Fe": 1.20,
    "Mg": 1.30,
    "Zn": 1.20,
    "Ca": 1.74,
    "Na": 1.54,
    "K": 1.96,
}

# 原子类型映射
atom_types = {
    # 骨架原子
    "N": "N",
    "CA": "C",
    "C": "C",
    "O": "O",
    "OXT": "O",
    # 常见侧链原子
    "CB": "C",
    "CG": "C",
    "CD": "C",
    "CE": "C",
    "CZ": "C",
    "OG": "O",
    "OG1": "O",
    "SG": "S",
    "SD": "S",
    "OD1": "O",
    "OD2": "O",
    "OE1": "O",
    "OE2": "O",
    "NE": "N",
    "NH1": "N",
    "NH2": "N",
    "NZ": "N",
    "ND1": "N",
    "ND2": "N",
    "NE2": "N",
    # 氢原子
    "H": "H",
    "HA": "H",
    "HB": "H",
    "HG": "H",
    "HD": "H",
}

ATOM_TYPES = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "CD",
    "CD1",
    "CD2",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "CZ",
    "CZ2",
    "CZ3",
    "CH2",
    "NE",
    "NE1",
    "NE2",
    "NH1",
    "NH2",
    "ND1",
    "ND2",
    "NZ",
    "OD1",
    "OD2",
    "OG",
    "OG1",
    "OE1",
    "OE2",
    "OH",
    "SD",
    "SG",
    "OXT",
]

RESIDUE_TYPES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

ATOM_ELEMENTS = {
    atom_type: elem
    for atom_type, elem in [
        ("N", "N"),
        ("CA", "C"),
        ("C", "C"),
        ("O", "O"),
        ("CB", "C"),
        ("CG", "C"),
        ("CG1", "C"),
        ("CG2", "C"),
        ("CD", "C"),
        ("CD1", "C"),
        ("CD2", "C"),
        ("CE", "C"),
        ("CE1", "C"),
        ("CE2", "C"),
        ("CE3", "C"),
        ("CZ", "C"),
        ("CZ2", "C"),
        ("CZ3", "C"),
        ("CH2", "C"),
        ("NE", "N"),
        ("NE1", "N"),
        ("NE2", "N"),
        ("NH1", "N"),
        ("NH2", "N"),
        ("ND1", "N"),
        ("ND2", "N"),
        ("NZ", "N"),
        ("OD1", "O"),
        ("OD2", "O"),
        ("OG", "O"),
        ("OG1", "O"),
        ("OE1", "O"),
        ("OE2", "O"),
        ("OH", "O"),
        ("SD", "S"),
        ("SG", "S"),
        ("OXT", "O"),
    ]
}