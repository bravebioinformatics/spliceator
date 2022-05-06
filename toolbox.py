#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created : 22/07/2020
Modified : 04/05/2021

@author: Scalzitti Nicolas 
"""


########################
#      Librairies      #
########################

import os
import csv


############################
#     One-hot Encoding     #
############################

# Conversion of nucleotide sequence in one-hot encoding sequence
def one_hot_encoding(sequence):
    
    sequence = sequence.upper()
    encoded_sequence = ""

    for nuc in sequence:
        if nuc == "A":
            encoded_sequence += "1000"
        elif nuc == "C":
            encoded_sequence += "0100"
        elif nuc == "G":
            encoded_sequence += "0010"
        elif nuc == "T":
            encoded_sequence += "0001"
        elif nuc == "X" or nuc == "N":
            encoded_sequence += "0000"

    return encoded_sequence

# Conversion of one-hot encoding sequence in nucleotide sequence
def one_hot_decoding(sequence):
    decoded_sequence = ""
    tpr = ""

    for nuc in sequence:
        tpr += nuc

        if tpr == "1000":
            decoded_sequence += "A"
            tpr = ""
        if tpr == "0100":
            decoded_sequence += "C"
            tpr = ""
        if tpr == "0010":
            decoded_sequence += "G"
            tpr = ""
        if tpr == "0001":
            decoded_sequence += "T"
            tpr = ""

    return decoded_sequence


################
#     Data     #
################

# Load data from a csv file. One column must contain target nucleotide sequences
def loading_data(file_path, size, column_seq=3, negative_dataset=False):
    """
    column_seq: the column number where are sequences
    """
    all_sequences = []

    try:
        with open(file_path, "r") as file_R1:
            for i, ligne in enumerate(file_R1):
                if negative_dataset and i == 0:
                    pass
                else:
                    ligne = ligne.strip().split(";")
                    sequence = ligne[column_seq].upper()
                    if size != 600:
                        sequence = sequence[300-int(size/2):300+int(size/2)]
                    if "N" in sequence or "Y" in sequence or "X" in sequence or "B" in sequence or 'M' in sequence or 'S' in sequence:
                        pass
                    else:                    
                        all_sequences.append(one_hot_encoding(sequence))
    except IndexError:
        pass

    #print(len(list(all_sequences)))
    return all_sequences

# Convert a negative nucleotide sequence into one-hot format sequence
def convert_negative_sequence(file_path, size, db_set, dataset_id, column_seq=2):    
    """
    file_path: path of the file
    size : size of input sequences
    db_set : type of negative dataset
    dataset_id : version of negative dataset
    column_seq: the column number where are sequences
    """
    loading_file_path_don = file_path + f"{db_set}/{dataset_id}/full/Raw/NEG_600_donor.csv"
    loading_file_path_acc = file_path + f"{db_set}/{dataset_id}/full/Raw/NEG_600_acceptor.csv"
    saving_file_path_don = file_path + f"{db_set}/{dataset_id}/full/Converted/NEG_converted_{size}_donor.txt"
    saving_file_path_acc = file_path + f"{db_set}/{dataset_id}/full/Converted/NEG_converted_{size}_acceptor.txt"

    os.makedirs(file_path + f"{db_set}/{dataset_id}/full/Converted/", exist_ok=True)

    with open(saving_file_path_don, "w") as file_W1:
        for seq in loading_data(loading_file_path_don, size, column_seq=2, negative_dataset=True):
            file_W1.write(seq + "\n")

    with open(saving_file_path_acc, "w") as file_W2:
        for seq in loading_data(loading_file_path_acc, size, column_seq=2, negative_dataset=True):
            file_W2.write(seq + "\n")

# Convert a positive nucleotid sequence into one-hot format sequence
def convert_positive_sequence(file_path, size, ss_type, dataset):
    set_sequence = set()

    if ss_type == "donor":
        loading_file_path = file_path + f"{dataset}/POS_donor_600.csv"
        saving_file_path  = file_path + f"{dataset}/Converted_2/POS_donor_{dataset.lower()}_{size}.txt"
        os.makedirs(file_path + f"{dataset}/Converted_2/", exist_ok=True)

    elif ss_type == "acceptor":
        loading_file_path = file_path + f"{dataset}/POS_acceptor_600.csv"
        saving_file_path  = file_path + f"{dataset}/Converted_2/POS_acceptor_{dataset.lower()}_{size}.txt"
        os.makedirs(file_path + f"{dataset}/Converted_2/", exist_ok=True)

    # Load the sequences from the target file and save the encoded sequences in a new file.
    with open(saving_file_path, "w") as file_W1:
        try:
            for seq in loading_data(loading_file_path,size, column_seq=1):
                if seq in set_sequence:
                    pass
                else:
                    file_W1.write(seq + "\n")
                    set_sequence.add(seq)
        except FileNotFoundError:
            print(f"[ERROR] File: {loading_file_path} not found")
            exit()

# Fusion of the negative set and the positive set (only one SS) in one file
def merged_positive_negative_set(current_dir, size, db_set, dataset_id, ss_type):
    """
    Create 2 files:
     -  Sequences_{size}.txt = contain all sequences (None, Donor or Acceptor)
     -  Labels_{size}.txt = contain labels for each sequence
    """
    saving_path = current_dir + f"/../Data/Datasets/Negative/{db_set}/{dataset_id}/full/Merged_2_{ss_type}/"
    os.makedirs(saving_path, exist_ok=True)

    # File containing all sequences
    with open(saving_path + f"Sequences_{size}.txt", "w") as file_W1:
        # File containing alls labels
        with open(saving_path + f"Labels_{size}.txt", "w") as file_W2:
            # Loading converted (one-hot) negative sequences
            with open(current_dir + f"/../Data/Datasets/Negative/{db_set}/{dataset_id}/full/Converted/NEG_converted_{size}_{ss_type}.txt", "r") as file_neg:
                sequences_neg_add=set()

                # Remove redundancy
                for ligne0 in file_neg:
                    sequences_neg_add.add(ligne0)

            # Write negative sequences in the Merged file
            for seq in sequences_neg_add:
                # All the sequences of the negative set have the label 0
                if len(seq) == (int(size)* 4) + 1: # +1 to take into account '\n'
                    file_W1.write(seq)
                    file_W2.write("0\n")
                else:
                    pass

            # Write positive Donor sequences in the Merged file
            if ss_type == "donor":
                if db_set == "All_0" or db_set == "All_1" or db_set == "All_2" or db_set == "All_10":
                    db_set = "All"
                if db_set == "GS_0" or db_set == "GS_1" or db_set == "GS_2" or db_set == "GS_10":
                    db_set = "GS"

                with open(current_dir + f"/../Data/Datasets/Positive/{db_set}/Converted/POS_donor_{db_set.lower()}_{size}.txt", "r") as file_don:
                    sequence_pos_don = set()

                    # Remove redundancy
                    for ligne1 in file_don:
                        sequence_pos_don.add(ligne1)
                    
                for seq in sequence_pos_don:
                    if seq in sequences_neg_add:
                        pass
                    else:
                        # All the sequences of the donor set have the label 1
                        if len(seq) == (int(size) * 4) + 1:
                            file_W1.write(seq)
                            file_W2.write("1\n")
                        else:
                            pass

            # Write positive Acceptor sequences in the Merged file
            elif ss_type == "acceptor":
                if db_set == "All_0" or db_set == "All_1" or db_set == "All_2" or db_set == "All_10":
                    db_set = "All"
                if db_set == "GS_0" or db_set == "GS_1" or db_set == "GS_2" or db_set == "GS_10":
                    db_set = "GS"

                with open(current_dir + f"/../Data/Datasets/Positive/{db_set}/Converted/POS_acceptor_{db_set.lower()}_{size}.txt", "r") as file_acc:
                    sequence_pos_acc = []
                
                    # Remove redundancy
                    for ligne2 in file_acc:
                        sequence_pos_acc.append(ligne2)

                for seq in sequence_pos_acc:
                    if seq in sequences_neg_add:
                        pass
                    else:
                        # All the sequences of the acceptor set have the label 1
                        if len(seq) == (int(size) * 4) + 1:
                            file_W1.write(seq)
                            file_W2.write("1\n")
                        else:
                            pass

def add_new_data(input_file, pos_neg, ss_type, quality, db_id, delimiter=";", canonical_only=False):
    """
    input_file: File containing new data
    pos_neg: 'positive' or 'negative'. Choose if you add positive or negative data. Input file contain for Positive file 4 rows (ID;Position;Sequence;Canonical) and 
    4 rows for negative data (ID;Position;Sequence;Type) (type is Intron, Exon or FP).
    ss_type: 'donor' or 'acceptor'.
    quality: 'HQ' or 'NF' quality of data, it is a high quality (HQ) sequence or not (NF)
    db_id: only if you add negativ data, choose the dataset id where you add new data.
    """

    list_to_add = []

    if pos_neg == "positive":
        with open(input_file, newline='') as file_R1:
            reader = csv.reader(file_R1, delimiter=delimiter)
            next(file_R1)
    
            for row in reader:
                ligne = ""
                try:
                    identifiant = row[0]
                except:
                    identifiant = "None"
                try:
                    position = int(row[1])
                except :
                    position = 0

                sequence = row[2]

                try:
                    canonical = bool(row[3])
                except:
                    canonical = True

                ligne += f'{identifiant};{position};{sequence};{canonical}\n' 
    
                if canonical_only:
                    if canonical == True:
                        list_to_add.append(ligne)
                    else:
                        pass
                else:
                    list_to_add.append(ligne)
    
        with open(os.getcwd() + f"/../Data/Datasets/Positive/Raw/Positive_{quality}_{ss_type}.csv", "a") as file_W1:
            for ligne in list_to_add:
                file_W1.write(ligne)
    
    elif pos_neg == "negative":
        with open(input_file, newline='') as file_R2:
            reader = csv.reader(file_R2, delimiter=delimiter)
            next(file_R2)
    
            for row in reader:
                ligne = ""
                try:
                    identifiant = row[0]
                except:
                    identifiant = "None"

                try:
                    position = int(row[1])
                except :
                    position = 0

                sequence = row[2]
                try:
                    neg_type = row[3]
                except:
                    neg_type = "Other"
                
                ligne += f'{identifiant};{position};{sequence};{neg_type}\n' 
    
                list_to_add.append(ligne)
    
        with open(os.getcwd() + f"/../Data/Datasets/{db_id}/full/Raw/NEG_400_{ss_type}.csv", "a") as file_W1:
            for ligne in list_to_add:
                file_W1.write(ligne)









