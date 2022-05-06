#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created : 21/07/2020
Modified : 04/06/2021

@author: Scalzitti Nicolas 
"""


########################
#      Librairies      #
########################

import os
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import toolbox as tb
from tqdm import tqdm
from statistics import mean
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import cv2
from pprint import pprint



########################
#         Data         #
########################

# Generate one-hot positive data from data files containing raw sequences and informations about SS
def generate_positive_data(current_dir):
    current_dir = current_dir + "/../Data/Datasets/Positive/"
    list_size = [20, 80, 140, 200, 400, 600]
    liste_dataset = ["All", "GS"]

    for db_set in liste_dataset:
        print("[INFO] ",db_set)
        for size in tqdm(list_size):
        	# Donor
            tb.convert_positive_sequence(current_dir, size, "donor", dataset=db_set)
            # Acceptor
            tb.convert_positive_sequence(current_dir, size, "acceptor", dataset=db_set)
    
    print("[INFO] Conversion of Positive Datasets")

# Generate one-hot negative data from data files containing raw sequences and informations about SS
def generate_negative_data(current_dir):
    current_dir = current_dir + f"/../Data/Datasets/Negative/"
    list_size = [20, 80, 140, 200, 400, 600]
    liste_dataset = ["All_0", "All_1", "All_2", "All_10", "GS_0", "GS_1", "GS_2", "GS_10"]
    liste_dataset_id = ["db_1","db_2","db_3","db_4","db_5","db_6","db_7","db_8","db_9","db_10"]

    for db_set in tqdm(liste_dataset):
        for dataset_id in liste_dataset_id:
            for size in list_size:
                print(f"[INFO] {db_set} - {dataset_id} - {size}")
                tb.convert_negative_sequence(current_dir, size, db_set, dataset_id)
    
    print("[INFO] Conversion of Negative Dataset")

# Fusion of the negative set and the positive set (only one SS) in one file
def merged_data(current_dir):
    list_size = [20, 80, 140, 200, 400, 600]
    liste_dataset = ["All_0", "All_1", "All_2", "All_10", "GS_0", "GS_1", "GS_2", "GS_10"]
    liste_dataset_id = ["db_1","db_2","db_3","db_4","db_5","db_6","db_7","db_8","db_9","db_10"]

    for db_set in tqdm(liste_dataset):
        for dataset_id in tqdm(liste_dataset_id):
            for size in tqdm(list_size):
                print(f"[INFO] {db_set} - {dataset_id} - {size}")
                tb.merged_positive_negative_set(current_dir, size, db_set, dataset_id, "donor")
                tb.merged_positive_negative_set(current_dir, size, db_set, dataset_id, "acceptor")

    print("[INFO] Merged files")

# generate data for two different models, one for each splice sites (Donor or Acceptor)
def generate_data(current_dir, positive_set, negative_set, merged):
    # Generate the positive dataset : raw data -> one-hot encoding data
    if positive_set:
        generate_positive_data(current_dir)

    # Generate the negative dataset (10 times) : raw data -> one-hot encoding data
    if negative_set:
        generate_negative_data(current_dir)
    
    # Fusion of the negative and positive dataset in one file, called 'Merged'
    if merged:
        print("[INFO] Starting merging")
        merged_data(current_dir)


#########################################
#         Training and test set         #
#########################################


# Extract one-hot encoding sequence from merged files and return one-hot sequences and labels
def load_data(sequences_file, labels_file):
    sequences = np.loadtxt(sequences_file, dtype='str')
    labels = np.loadtxt(labels_file, dtype='int')

    return sequences, labels

# Construction of the Training and Test set by separating the data randomly (80/20) 
def construct_initial_train_test_set(ss_type, test_size=0.2, new=False):
    list_size = [20, 80, 140, 200, 400, 600]
    liste_dataset = ["All_0", "All_1", "All_2", "All_10", "GS_0", "GS_1", "GS_2", "GS_10"]
    liste_dataset_id = ["db_1","db_2","db_3","db_4","db_5","db_6","db_7","db_8","db_9","db_10"]
    liste_ss = ["donor", "acceptor"]

    if new:
        data_dir = os.getcwd() + f"/../Data/Datasets/Negative/{db_set}/{dataset_id}/full/Merged_{ss_type}/"
        sequences, labels = load_data(data_dir + f"Sequences_{size}.txt", data_dir + f"Labels_{size}.txt")
        x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=float(test_size))
        save_test_train_set(size, db_set, dataset_id, ss_type, x_train, y_train, x_test, y_test)

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    else:
        for ss_type in tqdm(liste_ss):
            for db_set in tqdm(liste_dataset):
                for dataset_id in tqdm(liste_dataset_id):
                    for size in tqdm(list_size):
                        data_dir = os.getcwd() + f"/../Data/Datasets/Negative/{db_set}/{dataset_id}/full/Merged_{ss_type}/"
    
                        # Get sequences and their associated labels
                        sequences, labels = load_data(data_dir + f"Sequences_{size}.txt", data_dir + f"Labels_{size}.txt")
    
                        # Random spliting of the test and training sets
                        x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=float(test_size))
    
                        # Save the test and training sets (in order to reproduce the experiment)
                        save_test_train_set(size, db_set, dataset_id, ss_type, x_train, y_train, x_test, y_test)
    
    #return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

# Save the test set and the training set
def save_test_train_set(size, db_set, dataset_id, ss_type, x_train, y_train, x_test, y_test):
    
    # Create new directory
    savepath = os.getcwd() + f"/../Data/Train_test_set/{db_set}/{dataset_id}/"
    os.makedirs(savepath, exist_ok=True)
    
    nbr_tr = 0 # Number of sequences in the Training set
    nbr_te = 0 # Number of sequences in the Test set

    with open(savepath + f'Train_{ss_type}_{size}.csv', "w") as file_W1:
        # Training set
        for x, y in zip(x_train, y_train):
            encoding_seq = str(x).strip().replace(".", "").replace("[", "").replace("]","").replace(" ","").replace("\n","")
            
            file_W1.write(f"{nbr_tr};{tb.one_hot_decoding(encoding_seq)};{int(y)}\n")
            nbr_tr +=1

    with open(savepath + f'Test_{ss_type}_{size}.csv', "w") as file_W2:
        # Test set
        for x1, y1 in zip(x_test, y_test):
            encoding_seq1 = str(x1).strip().replace(".", "").replace("[", "").replace("]","").replace(" ","").replace("\n","")
            
            file_W2.write(f"{nbr_te};{tb.one_hot_decoding(encoding_seq1)};{int(y1)}\n")
            nbr_te +=1

# Loading of pre-built Test set and Training set in np.array format
def load_test_train_set(size, ss_type, db_set, dataset_id):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Loading of Train set
    with open(os.getcwd() + f"/../Data/Train_test_set/{db_set}/{dataset_id}/Train_{ss_type}_{size}.csv", "r") as file_train:
        for i, ligne in enumerate(file_train):
            ligne = ligne.strip().split(";")

            id_ligne = ligne[0]
            sequence = ligne[1].strip()
            label = int(ligne[2])

            encoding_sequence = tb.one_hot_encoding(sequence) # "00101000"

            es = encoding_sequence.replace(" ","")
            es = [int(i) for i in es] 
            es = np.array(es, dtype=int)

            x_train.append(es)
            y_train.append(label)

    # Loading of du Test set
    with open(os.getcwd() + f"/../Data/Train_test_set/{db_set}/{dataset_id}/Test_{ss_type}_{size}.csv", "r") as file_test:
        for i, ligne in enumerate(file_test):
            ligne = ligne.strip().split(";")

            id_ligne = ligne[0]
            sequence = ligne[1].strip()
            label = int(ligne[2])

            encoding_sequence = tb.one_hot_encoding(sequence) # "00101000"

            es = encoding_sequence.replace(" ","")
            es = [int(i) for i in es] 
            es = np.array(es, dtype=int)

            x_test.append(es)
            y_test.append(label)
    
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

# Main function to generate data. Converts raw sequences to one-hot format and then randomly generates training and test sets.
def main_data(current_dir, merged=True, positive_set=True, negative_set=True, train_test_build_only=True):
    liste_ss = ["donor", "acceptor"]

    # From the raw negative and positive (converted one-hot) sets, we generate the merged datasets.
    if train_test_build_only == False:
        generate_data(current_dir, positive_set=positive_set, negative_set=negative_set, merged=merged)
    
    # Randomly separates the data in train set (80%) and test set (20%) and saves the sequences in 2 files.
    for ss_type in liste_ss:
        construct_initial_train_test_set(ss_type, test_size=0.2)

        

#################################
#       Data informations       #
#################################


# Returns the number of positive and negative sequences in the initial dataset
def stat_merged_dataset():
    list_size = [20, 80, 140, 200, 400, 600]
    liste_dataset = ["All_0", "All_1", "All_2", "All_10", "GS_0", "GS_1", "GS_2", "GS_10"]
    liste_dataset_id = ["db_1","db_2","db_3","db_4","db_5","db_6","db_7","db_8","db_9","db_10"]
    liste_ss = ["donor", "acceptor"]

    with open(os.getcwd() + "/../Logs/Data_stats.log", "w") as file_W1:
        for ss_type in liste_ss:
            for db_set in liste_dataset:
                for dataset_id in liste_dataset_id:
                    for size in list_size:
        
                        dico = {}
                        with open(os.getcwd() + f"/../Data/Datasets/Negative/{db_set}/{dataset_id}/full/Merged_{ss_type}/Labels_{size}.txt", 'r') as file_R1:
                            for ligne in file_R1:
                                id_num = ligne.strip()
                    
                                if id_num in dico.keys():
                                    dico[id_num] +=1
                                else:
                                    dico[id_num] =1

                        file_W1.write(f"{db_set};{dataset_id};{ss_type};{size};")
                        file_W1.write(f"Negative;{dico['0']};Positive;{dico['1']}\n")

# Sequences in training set
def generate_training_sequence(size, ss_type, subset, dataset_id):
    liste_train = set()
    with open(os.getcwd() + f"/../Data/train_test_set/{dataset_id}/{subset}/train_{ss_type}_{size}.csv", "r") as file_train:
        for ligne in file_train:
            ligne = ligne.strip().split(";")
            seq = ligne[1]

            liste_train.add(seq)

    return liste_train

# Sequences in testing set
def generate_testing_sequence(size, ss_type, subset, dataset_id):
    liste_test = set()
    with open(os.getcwd() + f"/../Data/train_test_set/{dataset_id}/{subset}/test_{ss_type}_{size}.csv", "r") as file_test:
         for ligne in file_test:
            ligne = ligne.strip().split(";")
            seq = ligne[1]

            liste_test.add(seq)

    return liste_test

# check if we don't have identical sequences in test and training set
def check_unicity_of_train_test_set(size, ss_type, subset, dataset_id):
    # set allow to remove all redondant sequences
    liste_train = generate_training_sequence(size, ss_type, subset, dataset_id)
    liste_test  = generate_testing_sequence(size, ss_type, subset, dataset_id)
    liste_remove = []
    
    # extract sequences present in both lists
    liste_remove = liste_train & liste_test

    print(f"We have {len(liste_remove)} identical sequences in Testing and Train sets")

    return liste_remove

# Information about test/training set - nbr of label 0 and 1
def stat_train_test():
    list_size = [20, 80, 140, 200, 400, 600]
    liste_dataset = ["All_0", "All_1", "All_2", "All_10", "GS_0", "GS_1", "GS_2", "GS_10"]
    liste_dataset_id = ["db_1","db_2","db_3","db_4","db_5","db_6","db_7","db_8","db_9","db_10"]
    liste_ss = ["donor", "acceptor"]

    with open(os.getcwd() + "/../Logs/Train_test_stats.log", "w") as file_W1:
        for ss_type in liste_ss:
            for db_set in liste_dataset:
                for dataset_id in liste_dataset_id:
                    for size in list_size:
        
                        nbr_0 = 0
                        nbr_1 = 0
                        with open(os.getcwd() + f"/../Data/Train_test_set/{db_set}/{dataset_id}/Test_{ss_type}_{size}.csv", "r") as file_R1:
                            for ligne in file_R1:
                                ligne = ligne.strip().split(";")
                                label = ligne[2]
                    
                                if label == "0":
                                    nbr_0 += 1
                                elif label == "1":
                                    nbr_1 += 1
                    
                        nbr_00 = 0
                        nbr_11 = 0
                        with open(os.getcwd() + f"/../Data/Train_test_set/{db_set}/{dataset_id}/Train_{ss_type}_{size}.csv", "r") as file_R2:
                            for ligne in file_R2:
                                ligne = ligne.strip().split(";")
                                label = ligne[2]
                    
                                if label == "0":
                                    nbr_00 += 1
                                elif label == "1":
                                    nbr_11 += 1

                        file_W1.write(f"{db_set};{dataset_id};{ss_type};{size};")
                        file_W1.write(f"Test_set;0;{nbr_0};1;{nbr_1};Training_set;0;{nbr_00};1;{nbr_11}\n")

# Build a readable file containing informations about training and test sets
def organise_results_train_test_set():
    dico_donor = {}
    dico_acceptor = {}
    with open(os.getcwd() + "/../Logs/Train_test_stats.log", 'r') as file_R1:
        for ligne in file_R1:
            ligne = ligne.strip().split(";")

            subset = ligne[0]
            dataset_id = ligne[1]
            ss_type = ligne[2]
            size = ligne[3]
            test_true_site   = int(ligne[6])
            test_false_site  = int(ligne[8])
            train_true_site  = int(ligne[11])
            train_false_site = int(ligne[13])

            if ss_type == "donor":
                if subset not in dico_donor.keys():
                    dico_donor[subset] = {}
                    if dataset_id not in dico_donor[subset].keys():
                        dico_donor[subset][dataset_id] = [test_true_site,test_false_site,train_true_site,train_false_site]
                else:
                    if dataset_id not in dico_donor[subset].keys():
                        dico_donor[subset][dataset_id] = [test_true_site,test_false_site,train_true_site,train_false_site]
            elif ss_type == "acceptor":
                if subset not in dico_acceptor.keys():
                    dico_acceptor[subset] = {}
                    if dataset_id not in dico_acceptor[subset].keys():
                        dico_acceptor[subset][dataset_id] = [test_true_site,test_false_site,train_true_site,train_false_site]
                else:
                    if dataset_id not in dico_acceptor[subset].keys():
                        dico_acceptor[subset][dataset_id] = [test_true_site,test_false_site,train_true_site,train_false_site]

    with open(os.getcwd() + "/../Results/infos_train_test_sets2.csv", "w") as file_W1:
        file_W1.write('DONOR\n')

        for k,v in dico_donor.items():
            ligne_1 = str(k) + ",Test_neg,"
            ligne_2 = str(k) + ",Test_pos,"
            ligne_3 = str(k) + ",Train_neg,"
            ligne_4 = str(k) + ",Train_pos,"
            for kk, vv in dico_donor[k].items():
                ligne_1+=str(vv[0]) + ","
                ligne_2+=str(vv[1]) + ","
                ligne_3+=str(vv[2]) + ","
                ligne_4+=str(vv[3]) + ","
            file_W1.write(ligne_1 + "\n")
            file_W1.write(ligne_2 + "\n")
            file_W1.write(ligne_3 + "\n")
            file_W1.write(ligne_4 + "\n")

        file_W1.write('ACCEPTOR\n')

        for k,v in dico_acceptor.items():
            ligne_1 = str(k) + ",Test_neg,"
            ligne_2 = str(k) + ",Test_pos,"
            ligne_3 = str(k) + ",Train_neg,"
            ligne_4 = str(k) + ",Train_pos,"
            for kk, vv in dico_acceptor[k].items():
                ligne_1+=str(vv[0]) + ","
                ligne_2+=str(vv[1]) + ","
                ligne_3+=str(vv[2]) + ","
                ligne_4+=str(vv[3]) + ","
            file_W1.write(ligne_1 + "\n")
            file_W1.write(ligne_2 + "\n")
            file_W1.write(ligne_3 + "\n")
            file_W1.write(ligne_4 + "\n")




    exit()
    with open(os.getcwd() + "/../Results/infos_train_test_sets.csv", "w") as file_W1:
        file_W1.write("DONOR;Test_0;Test_1;Train_0;Train_1\n")
        for k, v in dico_donor.items():
            file_W1.write(k + "\n")
            for db_id, a in v.items():
                file_W1.write(f"{db_id};{a[0]};{a[1]};{a[2]};{a[3]}\n")
    
        file_W1.write("\nACCEPTOR;Test_0;Test_1;Train_0;Train_1\n")
        for k, v in dico_acceptor.items():
            file_W1.write(k + "\n")
            for db_id, a in v.items():
                file_W1.write(f"{db_id};{a[0]};{a[1]};{a[2]};{a[3]}\n")
        

########################
#        Models        #
########################


# CNN model architectures
def cnn_construction(size, filt, kernel_s, archi_model, seed, activation="relu", dropout_rate=0.2, neurons=100):
    if seed:
        tf.random.set_seed(351483773)
        print("[INFO] The seed is fixed")

    model = tf.keras.models.Sequential()

    # 1 layer + maxpooling
    if archi_model == 1:
        model.add(tf.keras.layers.Conv1D(filters=filt, kernel_size=kernel_s, strides=1, padding='same', batch_input_shape=(None, size, 4), activation="relu"))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(neurons, activation="relu"))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    # 2 layers + maxpooling
    elif archi_model == 2:
        model.add(tf.keras.layers.Conv1D(filters=filt, kernel_size=kernel_s, strides=1, padding='same', batch_input_shape=(None, size, 4), activation="relu"))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Conv1D(filters=filt*2, kernel_size=kernel_s - 1, strides=1, padding='same', batch_input_shape=(None, size, 4), activation="relu"))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(neurons, activation="relu"))
        model.add(tf.keras.layers.Dropout(dropout_rate))
     
    # 3 layers + maxpooling = BEST
    elif archi_model == 3:
        # Layer #1   #-1 pour ocuche 2 et 3
        model.add(tf.keras.layers.Conv1D(filters=filt, kernel_size=kernel_s, strides=1, padding='same', batch_input_shape=(None, size, 4), activation="relu"))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        # Layer #2
        model.add(tf.keras.layers.Conv1D(filters=filt*2, kernel_size=kernel_s - 2, strides=1, padding='same', batch_input_shape=(None, size, 4), activation="relu"))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
       
        # Layer #3
        model.add(tf.keras.layers.Conv1D(filters=filt*4, kernel_size=kernel_s - 4, strides=1, padding='same', batch_input_shape=(None, size, 4), activation="relu"))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(neurons, activation="relu"))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        
    # 4 layers + maxpooling
    elif archi_model == 4:
        model.add(tf.keras.layers.Conv1D(filters=filt, kernel_size=kernel_s, strides=1, padding='same', batch_input_shape=(None, size, 4), activation=activation))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Conv1D(filters=filt*2, kernel_size=kernel_s-1, strides=1, padding='same', batch_input_shape=(None, size, 4), activation=activation))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Conv1D(filters=filt*4, kernel_size=kernel_s-2, strides=1, padding='same', batch_input_shape=(None, size, 4), activation=activation))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Conv1D(filters=filt*8, kernel_size=kernel_s-3, strides=1, padding='same', batch_input_shape=(None, size, 4), activation=activation))
        model.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid'))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(neurons, activation="relu"))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Last layer
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model

# Main function to train the model
def train_cnn(current_dir, db_set, test_train_set, ss_type, dataset_id, size, epoch, filtre, taille_filtre, lr, archi_model, seed, batch_size,name_exp="", model_name="", callback_activate=False, plot=False, verbose=False, save=False, activation="relu", dropout_rate=0.3, neurons=90, opt="adamax"):
    # Generate new data in Train set and Test set
    
    if test_train_set == "new":
        construct_initial_train_test_set(ss_type, new=True)
        seq_train, label_train, seq_test, label_test = load_test_train_set(size, ss_type, db_set, dataset_id)

    # Load a pre-selection of data in the Training and Test sets.
    if test_train_set == "load":
        seq_train, label_train, seq_test, label_test = load_test_train_set(size, ss_type, db_set, dataset_id)

    if verbose:
        print(f"\nYou choose the subset {db_set}, - Dataset ID: {dataset_id}")
        print("Initial shape:\nSequence shapes:",seq_train.shape, "Label shapes:", label_train.shape)
        print("Example:\n",seq_train[10])

    # Data reshaping in 4 channels 
    seq_train = seq_train.reshape(-1, size, 4)
    seq_test = seq_test.reshape(-1, size, 4)
    
    if verbose:
        print()
        print("Reshaping = length/4:\nSequence shapes:",seq_train.shape, "Label shapes:",label_train.shape)
        print("Example:\n",seq_train[10])

    # Graphical representation of an one-hot input sequence
    #if plot:
    #    plot_input_sequence(seq_train, num=0)
    
    # Model construction
    model = cnn_construction(size, 
                             filtre, 
                             taille_filtre, 
                             archi_model,
                             seed,
                             activation=activation, # Activation function for each layer except the last one
                             dropout_rate=dropout_rate, 
                             neurons=neurons) #number of neurons in the last fully connected layer
                             
    # Print the architecture of the model
    if verbose:
        model.summary()

    # choice of optimizer, default=adamax
    if opt == "adamax":
        optimizer = tf.keras.optimizers.Adamax(lr)
    elif opt == "adam":
        optimizer = tf.keras.optimizers.Adam(lr)
    elif opt == "nadam":
        optimizer = tf.keras.optimizers.Nadam(lr)
    elif opt == "rms":
        optimizer = tf.keras.optimizers.RMSprop(lr)
    elif opt == "sgd":
        optimizer = tf.keras.optimizers.SGD(lr, momentum=0, nesterov=False)
    else:
        print("Please choose an optimizer in this list : adamax, adam, nadam, rms or sgd")
        exit()
    
    # Compilation of the model
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Directory where models are saved
    path_tmp = current_dir + "/../Models/tmp/"

    # Callbacks
    my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=5)], # Stop the training process when it no longer observes a progression at the end of X epochs
                    #tf.keras.callbacks.ModelCheckpoint(filepath= path_tmp + 'best_model.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=False), #Save the best model
                    #tf.keras.callbacks.TensorBoard(log_dir=current_dir + '/../Logs', histogram_freq=1, write_graph=True, write_images=True)] #Activate the tensorboard. Command line [tensorboard --log_dir DIRECTORY --bind_all]
    
    print("Start of fitting :")
    print(seq_train.shape)

    # Training process
    if callback_activate: 
        history = model.fit(seq_train, label_train, epochs=epoch, batch_size=batch_size, callbacks=my_callbacks, validation_split=0.15, verbose=1)
    else:
        history = model.fit(seq_train, label_train, epochs=epoch, batch_size=batch_size, validation_split=0.15, verbose=1) # or 0.25

    # Plot loss and accuracy curves
    #if plot:
    plot_results(dataset_id, ss_type, db_set, size, history, epoch, model_name)
    
    # Evaluation of the model in the test set
    loss, accuracy = model.evaluate(seq_test, label_test)
    print(f'\nAccuracy: {round(accuracy,3)}, Loss: {round(loss,3)}')

    # Predictions (probability) in the test set
    predictions = model.predict(seq_test)
    
    # Save results 
    if save:
        now = time.localtime(time.time())
        t = time.strftime("%y/%m/%d %H:%M", now)

        os.makedirs(current_dir + "/../Logs/", exist_ok=True)

        with open(current_dir + f"/../Logs/{name_exp}_Results_{ss_type}.log", "a") as file_W1:
            file_W1.write(f"{t};SS_type;{ss_type};Size;{size};Subset;{db_set};Dataset_ID;{dataset_id};Epoch;{epoch};Lr;{lr};Filtre;{filtre};Taille_filtre;{taille_filtre};Modele;{archi_model};")
            file_W1.write(f"Dropout;{dropout_rate};Optimizer;{opt};Neurons;{neurons};")
            file_W1.write(f"accuracy;{accuracy};Loss;{loss};Seed;{seed}\n")


    # Saving the model
    save_path = current_dir +  f"/../Models/{db_set}/{dataset_id}/"

    os.makedirs(save_path, exist_ok=True)

    if model_name == "":
        model.save(save_path + f"{ss_type}_{size}.h5")
    else:
        model.save(save_path + f"{model_name}.h5")
    print(f"[INFO] Model saved in: {save_path}")

    if callback_activate:
        print("\n>>> Callback is activated : You can activate tensorboard in a shell with : tensorboard --logdir Logs/ --bind_all\nPaste then the link in a browser")

    return model, predictions, seq_test, label_test, accuracy, loss

# Load a trained model
def load_trained_model(size, ss_type, subset, dataset_id, model_name, describe=False):
    try:
        model = tf.keras.models.load_model(os.getcwd() + f"/../Models/{subset}/{dataset_id}/{model_name}_{ss_type}_{size}.h5") 
    except:
        model = tf.keras.models.load_model(os.getcwd() + f"/../Models/{subset}/{dataset_id}/{model_name}{ss_type}_{size}.h5") 
    if describe:
        print(model.summary())

    return model

# Evaluation of a trained model, return also the predictions
def evaluate_trained_model(size, ss_type, subset, dataset_id, model_name):
    # Load the trained model
    model = load_trained_model(size, ss_type, subset, dataset_id, model_name)

    # Load the training and test set
    seq_train, label_train, seq_test, label_test = load_test_train_set(size, ss_type, subset, dataset_id) 

    seq = np.array(seq_test)
    labels = np.array(label_test)

    # Reshaping, with 4 channels 
    seq = seq.reshape(-1, size, 4)

    # Evaluation of the model on the test set sequences 
    loss, accuracy = model.evaluate(seq, labels)

    # Predictions (probability) in the test set
    predictions = model.predict(seq)

    matrix, accuracy, precision, sensitivity, specificity, f1_score = binary_confusion_matrix(predictions, labels)

    return matrix, accuracy, precision, sensitivity, specificity, f1_score


########################
#        Results       #
########################


def organize_results(ss_type):
    dico = {}
    with open(os.getcwd() + f"/../Logs/Spliceator_Results_{ss_type}.log", "r") as file_R1:
        for ligne in file_R1:
            ligne = ligne.strip().split(";")
            subset = ligne[6]
            size = ligne[4]
            id_db = ligne[8]
            accuracy = round(float(ligne[26])*100,2)

            if subset not in dico.keys():
                dico[subset] = {}

            if id_db not in dico[subset].keys():
                dico[subset][id_db] = {}

            if size not in dico[subset][id_db].keys():
                dico[subset][id_db][size] = accuracy

    #pprint(dico)
    with open(os.getcwd() + f"/../Results/Spliceator_Score_{ss_type}.csv", "w") as file_W1:
        for subset, v in dico.items():
            file_W1.write("\n" + subset + "\n")
            for elmt in dico[subset].items(): # All_1, All_2 etc...
                db_id = elmt[0]
                score = elmt[1]
                file_W1.write("\n" + db_id + ";")
                for x, y in score.items():
                    file_W1.write(str(y) + ";")

# Evaluate the performance of the model with computing the confusion matrix
def binary_confusion_matrix(predictions, label_test):
    TN = 0
    FP = 0
    FN = 0
    TP = 0

    for i_pred, j_true in zip(predictions, label_test):
        # None (Neg = 0)
        if int(j_true) == 0:
            if i_pred.argmax() == 0:
                TN +=1
            elif i_pred.argmax() == 1:
                FP += 1

        # SS (Pos=1)
        if int(j_true) == 1:
            if i_pred.argmax() == 0:
                FN +=1
            elif i_pred.argmax() == 1:
                TP += 1

    matrix = [[TN,FP],
              [FN,TP]]

    tp = matrix[1][1]
    tn = matrix[0][0]

    fp = matrix[0][1]
    fn = matrix[1][0]

    accuracy = round((TP + TN)/(TP+TN+FP+FN),5)
    precision = round(tp / (tp + fp),5)
    sensitivity = round(tp / (tp + fn),5)
    specificity = round(tn / (tn + fp),5)
    f1_score = round(2 * ((precision*sensitivity)/(precision+sensitivity)),5)

    print("[INFO] Accuracy:", accuracy, "\nPrecision:", precision, "\nSensitivity: ", sensitivity,"\nSpecificity: ", specificity, "\nF1 score:", f1_score)

    return matrix, accuracy, precision, sensitivity, specificity, f1_score


#######################
#    Explicability    #
#######################


# Generate a heatmap of the last convolution layer, with the most influential nucleotides.
def heatmap_activation_region(size, ss_type, subset, db_id, model_name,nbr_echantillon=1000):
    
    # Create output directory
    output = os.getcwd() + f"/../Results/Figures/Heatmap/"
    os.makedirs(output, exist_ok=True)    # source : https://medium.com/analytics-vidhya/visualizing-activation-heatmaps-using-tensorflow-5bdba018f759
    
    # Loading of Training and test set
    seq_train, label_train, seq_test, label_test = load_test_train_set(size, ss_type, subset, db_id)
    
    # Average matrix for all results
    all_matrix_ss = []
    all_matrix_no_ss = []

    # Loading the model
    model = load_trained_model(size, ss_type, subset, db_id, model_name)
    #print(model.summary())

    # Recover the name of the last layer
    if ss_type == "donor":
        layer = model.layers[6]._name
    elif ss_type == "acceptor":
        layer = model.layers[6]._name

    # Evaluate X number of sample
    for i in tqdm(range(nbr_echantillon)): #(len(seq_train)):
        i = int(i)
        x = seq_train[i] # shape = (80,)
        x = x.reshape(-1, size, 4)
        x = x.astype('float64') 
        label = label_train[i]
        
        preds = model.predict(x)
    
        with tf.GradientTape() as tape:
            last_conv_layer = model.get_layer(layer) # Last conv layer
            iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
            model_out, last_conv_layer = iterate(x)
            class_out = model_out[:, np.argmax(model_out[0])]
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
      
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        if heatmap[0][0] < 0.0:
            heatmap = heatmap*-1
    
        heatmap = np.maximum(heatmap, 0) # Compare two arrays and returns a new array containing the element-wise maxima, but here we have only one array
        heatmap /= np.max(heatmap) # divide by
    
        # display the heatmap
        #plt.matshow(heatmap)
        #plt.show()

        heatmap = cv2.resize(heatmap, (x.shape[1], x.shape[0]*5))
    
        plt.matshow(heatmap)#, cmap="PuBu") #bwr
        plt.colorbar() # Legend

        plt.yticks([])
        plt.xlabel("Position", fontsize=13)

        if int(label) == 0:
            label = f"none_{ss_type}"
            all_matrix_no_ss.append(heatmap)
        elif int(label) == 1:
            label = ss_type
            all_matrix_ss.append(heatmap)

        plt.close()

    #plt.savefig(output + f"Heatmap_{str(i)}_{label}.png")
  
    all_matrix_no_ss = np.mean(np.array(all_matrix_no_ss), axis=0)
    all_matrix_ss = np.mean(np.array(all_matrix_ss), axis=0)

    plt.matshow(all_matrix_no_ss)#, cmap="PuBu")

    plt.yticks([])
    plt.xlabel("Position", fontsize=13)
    plt.savefig(output + f"/../Heatmap_mean_no_ss_{ss_type}_{db_id}.png")
    plt.close()

    plt.matshow(all_matrix_ss)#, cmap="PuBu")
    plt.yticks([])
    plt.xlabel("Position", fontsize=13)
    plt.savefig(output + f"/../Heatmap_mean_ss_{ss_type}_{db_id}.png")
    plt.show()
    plt.close()


#######################
#      Graphical      #
#######################


# Displays the progress curve of accuracy and loss.
def plot_results(dataset_id, ss_type, db_set, size,history, epoch, model_name):
    loss_curve = history.history["loss"]
    acc_curve = history.history["accuracy"]
    val_loss_curve = history.history["val_loss"]
    val_acc_curve = history.history["val_accuracy"]

    plt.plot(loss_curve, label="Loss", color="#ffb8b8")
    plt.plot(acc_curve, label="Accuracy", color="#ff9f1a")
    plt.plot(val_loss_curve, label="val_loss", color="#ff3838")
    plt.plot(val_acc_curve, label="val_Accuracy", color="#f6e58d")

    plt.xlabel('Epoch')
    plt.xlim(0, int(epoch))
    plt.legend()

    os.makedirs(os.getcwd() + f"/../Results/Figures/{db_set}/{dataset_id}/", exist_ok=True)
    plt.savefig(os.getcwd() + f"/../Results/Figures/{db_set}/{dataset_id}/{model_name}.png", dpi=300)
    
    plt.close()
    #plt.show()

# Graphical representation of on input sequence
def plot_input_sequence(seq_train, num=0):
    plt.figure()
    plt.imshow(seq_train[num])
    plt.grid(False)
    plt.show()


########################
#         Other        #
########################


# Check the version of Tensorflow
def check_version():
    print("[INFO] : TF Version: ", tf.__version__)

    tpr = str(tf.__version__).split(".")
    version = tpr[0]
    release = tpr[1]

    if tf.__version__ != "2.4.1":
        if version < 2:
            print(f"[ERROR] please use the version 2 of Tensorflow")
        if version == 2 and release < 4:
            print(f"Your version of TensorFlow ({tf.__version__}) is incompatible with this program, please upgrade to version 2.4.1 (or use the virtual environment)")
        exit()
    else:
        print("[INFO] We cannot guarantee that the program will work in the next Tensorflow update, but we will do what is necessary.\nHowever, it is possible to use the virtual environment available in git to have a stable version of Spliceator")


def start_data_creation(current_dir, merged=False, positive_set=False, negative_set=False, train_test_build_only=False):
    print("[INFO] Start data generation")
    main_data(current_dir, merged=merged, positive_set=positive_set, negative_set=negative_set, train_test_build_only=train_test_build_only)

def start_info():
    print("[INFO] Information about the datasets")
    os.makedirs(current_dir + "/../Logs/", exist_ok=True)
    #stat_merged_dataset()
    stat_train_test()
    print("[INFO] Files 'Data_stats.log' and 'Train_test_stats.log' created")

def start_evaluation(size, dataset_id, subset, model_name):
    # Donor
    print("[INFO] DONOR")
    matrix, accuracy, precision, sensitivity, specificity, f1_score = evaluate_trained_model(size, 'donor', subset, dataset_id, model_name)
    donor = [matrix, accuracy, precision, sensitivity, specificity, f1_score]
    # Acceptor
    print("[INFO] ACCEPTOR")
    matrix, accuracy, precision, sensitivity, specificity, f1_score = evaluate_trained_model(size, 'acceptor', subset, dataset_id, model_name)
    acceptor = [matrix, accuracy, precision, sensitivity, specificity, f1_score]
    
    return donor, acceptor


if __name__ == '__main__':

    current_dir = os.getcwd()

    ##### INFOS #####
    print("[INFO] Spliceator version 2.1")
    print("[INFO] Activate the virtual environment")
    print("[AUTHOR] Scalzitti Nicolas")
    print("[TEAM] ICube Laboratory UMR7357 - CSTB Strasbourg (France)\n")
    ##### INFOS #####

    #####  WORKING PARAMETERS #####
    data_creation        = False     # If True, generate a training and test sets
    model_training       = True     # If True, run start the training of a new model (based on data in dataset_id)
    info                 = False    # Get some informations about data and dataset
    model_evaluation     = False    # Evalute the model with the test set
    #####  WORKING PARAMETERS #####


    ##### DATA PARAMETERS #####
    size = 400          # (400, 200, 140, 80, 20) - Length of genomic sequences
    dataset_id = "db_1"  # The name of the dataset using to train and test the model (in Data/Datasets)
    subset = "GS_1"     # The subset of data, containing different ratio of positive and negative data
    ss_type = "donor"    # 'donor' or 'acceptor': choose one if you select 2 models (if you select 1 model keep the default value)
    specie = "Human"     # Specie for the benchmark evaluation choice : Human, Worm, Fly, Danio, Thaliana (Check benchmark.py)
    name_exp = "LSTM"
    current_dir = os.getcwd()
    ##### DATA PARAMETERS #####

    list_size = [20,80,140,200,400]
    list_dataset = ["db_1","db_2","db_3","db_4","db_5","db_6","db_7","db_8","db_9","db_10"]
    list_subset =  ["GS_0", "GS_1", "GS_2", "GS_10","All_0", "All_1", "All_2","All_10"]
    # Generation of new training and test sets (based on the data in dataset_id)

    if data_creation:
        start_data_creation(current_dir, merged=True, positive_set=False, negative_set=False, train_test_build_only=False)
        """
        If merged is True: Merged positive and negative one-hot encodinbg data in one file with sequences + one file with labels
        If positive_set is True: Convert raw positive data to one-hot encoding data
        If negative_set is True: Convert raw negative data to one-hot encoding data
        If train_test_build_only is True: deactivate the data generation and build only train and test sets
        """

    # Training process
    if model_training:
        #seed = 301
        #tf.random.set_seed(seed)

        for subset in list_subset:
            for dataset_id in list_dataset:
                for size in list_size:

                    model, prediction, sequence, label, accuracy, loss = train_cnn(
                                                            current_dir       = current_dir, # working path
                                                            db_set            = subset,      # the subset used to train and test the model (do not modify this)
                                                            test_train_set    = "load",      # 'load' to load a pre-built Training and Test sets, or 'new' if you want to create a new Training and Test sets from an existing dataset
                                                            ss_type           = ss_type,     # if number_model = 1, this is not taken into account
                                                            dataset_id        = dataset_id,  # The Ith dataset
                                                            size              = size,        # size of the input sequences
                                                            epoch             = 400,         # ~500
                                                            filtre            = 16,          # >10 default 16
                                                            taille_filtre     = 7,           # Kernel Size #default 7
                                                            lr                = 1e-5,        # or 1e-4
                                                            archi_model       = 3,           # Best = 3 (3 conv. layers)
                                                            seed              = False,       # fix a seed (True or false)
                                                            batch_size        = 32,          # Tha batch size
                                                            name_exp          = name_exp,    # Name of the experience can be ""
                                                            model_name        = f"{name_exp}_{ss_type}_{size}", # Name of the saved trained model 
                                                            callback_activate = False,       # Activate or deactivate the callback process
                                                            activation        = "relu",      # Activation function for each layer except the last one
                                                            dropout_rate      = 0.2,         # 0.2-0.3
                                                            neurons           = 100,         # ~90, number of neurons in the last fully connected layer
                                                            opt               = "adamax",    # type of the optimizer
                                                            plot              = True,        # to plot the training curve
                                                            verbose           = False,       # display some informations
                                                            save              = True)        # save some informations in log
                                                            
    # Informations about data
    if info:
        start_info()
        
    # Trained model evaluation ##
    if model_evaluation:

        list_size = [20,80,140,200,400,600]
        list_dataset = ["db_1","db_2","db_3","db_4","db_5","db_6","db_7","db_8","db_9","db_10"]
        list_subset =  ["All_0", "All_1", "All_2","All_10", "GS_0", "GS_1", "GS_2", "GS_10"]
        list_ss = ["donor", "acceptor"]

        with open(os.getcwd() + "/../Results/Results_benchmarks_test.csv", "w") as file_W1:
            header = "SS type;Subset;DB_ID;Size;Accuracy;Precision;Sensitivity;Specificity;F1 score"
            file_W1.write(header + "\n")

            for subset in list_subset:
                for dataset_id in list_dataset:
                    donor, acceptor = start_evaluation(200, dataset_id, subset, "")
                    file_W1.write(f"Donor;{subset};{dataset_id};200;{donor[1]};{donor[2]};{donor[3]};{donor[4]};{donor[5]}\n")
                    file_W1.write(f"Acceptor;{subset};{dataset_id};200;{acceptor[1]};{acceptor[2]};{acceptor[3]};{acceptor[4]};{acceptor[5]}\n")
        

    if data_creation is False and data_creation is False and model_training is False and info is False and model_evaluation is False and benchmark_evaluation is False:
        print("\n[INFO] All Working Parameters are False, please choose at least one on True")
        exit()