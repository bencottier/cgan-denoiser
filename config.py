# -*- coding: utf-8 -*-
"""
config.py

Parameters for different models

author: Ben Cottier (git: bencottier)
"""
from os.path import join

class Config:
    """
    Configuration parameters for the Conditional GAN
    """
    # Dimensions
    raw_size = 256
    adjust_size = 256
    train_size = 256
    channels = 1
    base_number_of_filters = 64
    kernel_size = (3, 3)
    strides = (2, 2)

    # Fixed model parameters
    leak = 0.2
    dropout_rate = 0.5

    # Hyperparameters
    learning_rate = 2e-4
    beta1 = 0.5
    L1_lambda = 100

    # Data
    buffer_size = 1425
    batch_size = 5
    max_epoch = 20
    max_training_cases = 1140
    validation_split = 0.2
    test_cases = [
        0,    6,   17,   29,   30,   33,   34,   49,   51,   52,   66,
        68,   70,   82,   83,   94,   96,   99,  101,  104,  107,  114,
        118,  121,  123,  130,  135,  141,  151,  152,  159,  160,  172,
        182,  185,  191,  202,  204,  206,  211,  221,  228,  232,  233,
        242,  244,  247,  249,  255,  260,  270,  293,  299,  301,  305,
        311,  315,  330,  331,  333,  334,  336,  353,  357,  362,  368,
        377,  383,  387,  392,  395,  396,  399,  410,  412,  416,  419,
        429,  430,  435,  451,  454,  456,  461,  463,  464,  468,  473,
        477,  481,  486,  488,  489,  493,  497,  524,  525,  546,  560,
        565,  567,  572,  579,  588,  589,  597,  599,  610,  615,  620,
        626,  627,  633,  640,  645,  647,  652,  662,  664,  665,  670,
        675,  678,  685,  687,  690,  691,  693,  704,  706,  709,  712,
        713,  714,  716,  717,  718,  724,  726,  734,  736,  750,  755,
        771,  773,  775,  786,  804,  814,  815,  817,  824,  828,  831,
        834,  842,  845,  847,  854,  868,  874,  883,  899,  905,  907,
        909,  911,  912,  915,  916,  923,  924,  930,  933,  934,  940,
        943,  947,  950,  951,  952,  960,  962,  975,  981,  988,  990,
        991, 1008, 1011, 1012, 1019, 1022, 1028, 1029, 1032, 1033, 1045,
        1049, 1052, 1056, 1057, 1058, 1059, 1065, 1068, 1074, 1086, 1092,
        1098, 1099, 1102, 1109, 1110, 1113, 1114, 1121, 1127, 1128, 1129,
        1131, 1133, 1134, 1140, 1141, 1151, 1159, 1166, 1187, 1188, 1204,
        1206, 1207, 1209, 1211, 1216, 1230, 1232, 1233, 1234, 1235, 1238,
        1245, 1248, 1259, 1262, 1271, 1273, 1275, 1276, 1290, 1297, 1298,
        1299, 1302, 1307, 1311, 1316, 1317, 1319, 1324, 1326, 1328, 1330,
        1331, 1335, 1336, 1339, 1343, 1346, 1356, 1361, 1363, 1366, 1367,
        1370, 1373, 1379, 1385, 1386, 1390, 1398, 1408, 1413, 1415]

    # Data storage
    save_per_epoch = max_epoch
    exp_name = 'fractal_oasis1_cgan'
    data_path = 'data/'
    # root_path = "/home/Student/s4360417/honours/datasets/oasis1/"  # TODO
    root_path = "/home/ben/projects/honours/datasets/oasis1/"  # TODO
    input_path = root_path + "slices_artefact/"
    label_path = root_path + "slices_pad/"
    model_path = join('out', exp_name, 'model')
    results_path = join('out', exp_name, 'results')
