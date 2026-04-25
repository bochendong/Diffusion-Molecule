"""Shared table schema for the PhysTabMol prototype."""

TARGET_COLUMNS = ["MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB", "SA"]

COUNT_COLUMNS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "ring_count"]

GROUP_COLUMNS = [
    "fg_ester",
    "fg_amide",
    "fg_amine",
    "fg_alcohol",
    "fg_halogen",
]

TABLE_COLUMNS = TARGET_COLUMNS + COUNT_COLUMNS + ["scaffold_class"] + GROUP_COLUMNS

INTEGER_COLUMNS = set(COUNT_COLUMNS + ["scaffold_class"] + GROUP_COLUMNS)

BOUNDS = {
    "MW": (80.0, 650.0),
    "LogP": (-3.0, 7.0),
    "QED": (0.0, 1.0),
    "TPSA": (0.0, 180.0),
    "HBD": (0.0, 6.0),
    "HBA": (0.0, 12.0),
    "RB": (0.0, 15.0),
    "SA": (1.0, 8.0),
    "C": (1.0, 45.0),
    "N": (0.0, 8.0),
    "O": (0.0, 12.0),
    "S": (0.0, 3.0),
    "F": (0.0, 4.0),
    "Cl": (0.0, 3.0),
    "Br": (0.0, 2.0),
    "I": (0.0, 1.0),
    "ring_count": (0.0, 5.0),
    "scaffold_class": (0.0, 4.0),
    "fg_ester": (0.0, 3.0),
    "fg_amide": (0.0, 3.0),
    "fg_amine": (0.0, 3.0),
    "fg_alcohol": (0.0, 4.0),
    "fg_halogen": (0.0, 4.0),
}

