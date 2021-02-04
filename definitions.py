bacteria_list = {
    # From MRSA 1 to S. lugdunensis
    16: 'MRSA 1',
    17: 'MRSA 2',
    14: 'MSSA 1',
    18: 'MSSA 2',
    15: 'MSSA 3',
    20: 'S. epidermidis',
    21: 'S. lugdunensis',

    # From Group A Strep. to Group G Strep.
    26: 'Group A Strep.',
    27: 'Group B Strep.',
    28: 'Group C Strep.',
    29: 'Group G Strep.',
    # S. sanguinis,
    25: 'S. sanguinis',

    # E. faecalis 1 + E. faecalis 2
    6: 'E. faecalis 1',
    7: 'E. faecalis 2',

    # S. enterica
    19: 'S. enterica',

    # # Ignored classes
    # 0: 'C. albicans',
    # 1: 'C. glabrata',
    # 2: 'K. aerogenes',
    # 3: 'E. coli 1',
    # 4: 'E. coli 2',
    # 5: 'E. faecium',
    # 8: 'E. cloacae',
    # 9: 'K. pneumoniae 1',
    # 10: 'K. pneumoniae 2',
    # 11: 'P. mirabilis',
    # 12: 'P. aeruginosa 1',
    # 13: 'P. aeruginosa 2',
    # 22: 'S. marcescens',
    # 23: 'S. pneumoniae 2',
    # 24: 'S. pneumoniae 1',
}

antibiotics = {
    0: 'Meropenem',
    1: 'Ciprofloxacin',
    2: 'TZP',
    3: 'Vancomycin',
    4: 'Ceftriaxone',
    5: 'Penicillin',
    6: 'Daptomycin',
    7: 'Caspofungin',
}

bacteria_antibiotics = {
    3: 0,
    4: 0,
    9: 0,
    10: 0,
    2: 0,
    8: 0,
    11: 0,
    22: 0,

    19: 1,

    12: 2,
    13: 2,

    14: 3,
    18: 3,
    15: 3,
    20: 3,
    21: 3,
    16: 3,
    17: 3,

    23: 4,
    24: 4,

    26: 5,
    27: 5,
    28: 5,
    29: 5,
    25: 5,
    6: 5,
    7: 5,

    5: 6,

    0: 7,
    1: 7
}


def bacteria_name(index):
    return bacteria_list[index]


def bacteria_class(name):
    for key, value in bacteria_list.items():
        if value == name:
            return key
