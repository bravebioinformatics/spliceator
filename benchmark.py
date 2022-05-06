#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created : 21/07/2020
Modified : 30/09/2020

@author: Scalzitti Nicolas 
"""

# Run the evaluation on the benchmark

import os
import random
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import toolbox as tb
import time
import pandas as pd


def accuracy_(TP, TN, all_):
    acc = ((TP + TN)/all_)
    return round(acc*100,2)

def precision_(TP, FP):
    pre = (TP / (TP + FP))
    return round(pre*100,2)

def sensitivity_(TP, FN):
    sn = (TP / (TP + FN))
    return round(sn*100,2)

def specificity_(TN, FP):
    sn = (TN / (TN + FP))
    return round(sn*100,2)

def f1_score_(precision, sensitivity):
    f1 = (2 *((precision * sensitivity) / (precision + sensitivity)))
    return round(f1*100,2)

# Load data
def load_data(sequences, labels):
    sequences = np.loadtxt(sequences, dtype='str')
    labels = np.loadtxt(labels, dtype='int')

    return sequences, labels

def create_df():
    return pd.read_csv(os.getcwd() + f"/../Results/Quality_Benchmark_Spliceator.csv", sep=';', names=["Programm", "Specie",  "SS_type", "Size", "Subset","Set","Db_id","Accuracy", "Precision" ,"Sensibility", "Specificity", "F1_score"])

# Load a trained model
def load_trained_model(size, subset, dataset_id, ss_type, soft, model_name=""): # High_kernel Archi2
    if soft == "spliceator":
        if model_name == "":
            name = f"{ss_type}_{size}"
        else:
            name = f"{model_name}_{ss_type}_{size}"
        model = tf.keras.models.load_model(os.getcwd() + f"/../Models/{subset}/{dataset_id}/{name}.h5")

    if soft == "splicefinder":
        model_name = "SF_100"
        model = tf.keras.models.load_model(os.getcwd() + f"/../Models/{model_name}.h5")

    #print(model.summary())
    return model, model_name

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

    #print("[INFO] Matrix\n", matrix)
    TP = matrix[1][1]
    TN = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    all_ = TP + TN + FP + FN

    print("FN", FN, "FP", FP, "TP", TP, "TN", TN)

    accuracy =    accuracy_(TP, TN, all_)
    precision =   precision_(TP, FP) 
    sensitivity = sensitivity_(TP, FN) 
    specificity = specificity_(TN, FP) 
    f1_score =    round((2 * ((precision*sensitivity)/(precision+sensitivity))),2)

    print(f"[INFO] Accuracy - Precision - Sensitivity - Specificity - F1_score")
    print(f"{accuracy},,{precision},,{sensitivity},,{specificity},,{f1_score}")
    
    results = [matrix, accuracy, precision, sensitivity, specificity, f1_score]
    return results

#Matrice with 3 classes
def matrice_confusion_one_model(predictions, label_test):
    # Variable initializations
    rien_rien = 0
    rien_don = 0
    rien_acc = 0
    
    donor_rien = 0
    donor_donor = 0
    donor_acc = 0

    acc_rien = 0
    acc_don = 0
    acc_acc = 0

    for i_pred, j_true in zip(predictions, label_test):

        #Conversion of one-hot labels to normal format
        if j_true[0] == 1.0:   # Acceptor
            j_true = 0
        elif j_true[1] == 1.0: # Donor
            j_true = 1 
        elif j_true[2] == 1.0: # None
            j_true = 2

        # Donor
        if int(j_true) == 1:
            if  i_pred.argmax() == 1:  # Donor VS Donor
                donor_donor +=1
            elif i_pred.argmax() == 0: # Donor VS Acceptor
                donor_acc += 1
            elif i_pred.argmax() == 2: # Donor VS None
                donor_rien += 1

        # Acceptor
        if int(j_true) == 0:
            if  i_pred.argmax() == 1:  # Acceptor VS Donor
                acc_don +=1
            elif i_pred.argmax() == 0: # Acceptor VS Acceptor
                acc_acc += 1
            elif i_pred.argmax() == 2: # Acceptor VS None
                acc_rien +=1

        # None
        if int(j_true) == 2:
            if  i_pred.argmax() == 1:   # None VS Donor
                rien_don +=1
            elif  i_pred.argmax() == 0: # None VS Acceptor
                rien_don +=1
            elif i_pred.argmax() == 2: # None VS None
                rien_rien +=1

   #   True        0          1            2       
    matrix = [ [acc_acc   ,acc_don     ,acc_rien],     # 0
               [donor_acc ,donor_donor ,donor_rien],   # 1        Predict
               [rien_acc  ,rien_don    ,rien_rien] ]   # 2 
    print(matrix)

    # Acceptor
    tp = matrix[0][0]
    tn = matrix[1][1]+matrix[1][2]+matrix[2][1]+matrix[2][2]
    fp = matrix[0][2]+matrix[0][1]
    fn = matrix[2][0]+matrix[1][0]

    precision_donor = round(tp / (tp + fp),3)
    recall_donor = round(tp / (tp + fn),3)
    specificity_donor = round(tn / (tn + fp),3)
    accuracy_donor = round( (tp+tn)/(tp + tn + fp + fn), 3)
    f1_score_donor = 2 * ((precision_donor * recall_donor) / (precision_donor + recall_donor))
    print("Acceptor - Accuracy: ", accuracy_donor ,"Precision: ", precision_donor,"Sn: ", recall_donor, "Specificity: ", specificity_donor)
    print("F1 Score Acceptor:", f1_score_donor)
    # Donor
    tp = matrix[1][1]
    tn = matrix[0][0]+matrix[0][2]+matrix[2][0]+matrix[2][2]
    fp = matrix[1][0]+matrix[1][2]
    fn = matrix[0][1]+matrix[2][1]

    precision_acc = round(tp / (tp + fp),3)
    recall_acc = round(tp / (tp + fn),3)
    specificity_acc = round(tn / (tn + fp),3)
    accuracy_acc = round( (tp+tn)/(tp + tn + fp + fn), 3)
    f1_score_acc = 2 * ((precision_acc * recall_acc) / (precision_acc + recall_acc))

    print("Donor - Accuracy: ", accuracy_acc, "Precision: ",precision_acc,"Sn: ", recall_acc, "Specificity: ", specificity_acc)
    print("F1 Score Donor:", f1_score_acc)

    #None
    tp = matrix[2][2]
    tn = matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1]
    fp = matrix[2][0]+matrix[2][1]
    fn = matrix[0][2]+matrix[1][2]

    precision_rien = round(tp / (tp + fp),3)
    recall_rien = round(tp / (tp + fn),3)
    specificity_rien = round(tn / (tn + fp),3)
    accuracy_rien = round( (tp+tn)/(tp + tn + fp + fn), 3)
    f1_score_rien = 2 * ((precision_rien * recall_rien) / (precision_rien + recall_rien))

    print("None - Accuracy: ",accuracy_rien ,"Precision:",precision_rien,"Sn: ", recall_rien, "Specificity: ", specificity_rien)
    print("F1 Score rien:", f1_score_rien)

    # All model
    tp_all = matrix[2][0] + matrix[1][1] + matrix[0][2]
    fp_all = (matrix[2][2]+matrix[2][1]) + (matrix[1][0]+matrix[1][2]) + (matrix[0][0]+matrix[0][1])
    fn_all = (matrix[0][0]+matrix[1][0]) + (matrix[0][1]+matrix[2][1]) + (matrix[1][2]+matrix[2][2])
    tn_all = (matrix[0][0]+matrix[0][2]+matrix[2][0]+matrix[2][2]) + (matrix[1][0]+matrix[1][1]+matrix[2][0]+matrix[2][1]) + (matrix[1][0]+matrix[1][1]+matrix[2][0]+matrix[2][1])
    precision_all = round(tp_all / (tp_all + fp_all),3)
    recall_all = round(tp_all / (tp_all + fn_all),3)    
    specificity_all = round(tn_all / (tn_all + fp_all),3)

# Count the number of canonical and non-canoncial SS
def count_cano(params):
    params = params

    sequences, labels = load_data(os.getcwd() + f"/../Data/Benchmarks/{params['specie']}/SA_sequences_{params['ss_type']}_400_{params['num']}.fasta", # Sequences
                                   os.getcwd() + f"/../Data/Benchmarks/{params['specie']}/SA_labels_{params['ss_type']}_400_{params['num']}.fasta")    # Labels

    cano = 0
    nc = 0
    if params['ss_type'] == "donor":
        for i,seq in enumerate(sequences):
            seq = seq[200 - int(params["size"]/2):200 + int(params["size"]/2)]

            if i < 10000:
                pass
            else:

                if params['ss_type'] == "acceptor":
                        
                        if seq[100] == 'A'and seq[101] == 'g':
                            cano +=1
                        else:
                            nc +=1
                if params['ss_type'] == "donor":
                    #print(seq)
                    
                    if seq[100] == 'G'and seq[101] == 'T':
                        cano +=1
                    else:
                        nc +=1

    print(params['specie'],"Total:", len(sequences), "canonical:", cano, 'non-canoncial:',nc, "All:", cano+nc )

def evaluate_spliceator_or_splicefinder(params):
    params = params

    if params['soft'] == "spliceator":
        model, model_name = load_trained_model(params["size"], params["subset"], params["db_id"], params["ss_type"], "spliceator", params['model_name'])
        sequences, labels = load_data(os.getcwd() + f"/../Data/Benchmarks/{params['specie']}/SA_sequences_{params['ss_type']}_400_{params['num']}.fasta", # Sequences
                                      os.getcwd() + f"/../Data/Benchmarks/{params['specie']}/SA_labels_{params['ss_type']}_400_{params['num']}.fasta")    # Labels
    elif params['soft'] == "splicefinder":

        model, _ = load_trained_model(params["size"], params["subset"], params["db_id"], params["ss_type"], "splicefinder")
        sequences, labels = load_data(os.getcwd() + f"/../Data/Benchmarks/{params['specie']}/SF_sequences_400_{params['num']}.fasta", # Sequences
                                      os.getcwd() + f"/../Data/Benchmarks/{params['specie']}/SF_labels_400_{params['num']}.fasta")    # Labels
        print(sequences.shape, labels.shape)
        
    all_converted_sequences = []

    if params['specie'] == "Protist":  
        # Encoding data
        if params['soft'] == "spliceator":

            print(f"[INFO] Data encoding for Spliceator")
            for i,seq in enumerate(sequences):
    
                if params['ss_type'] == "acceptor":
                    seq = seq[3:-1] + "N"
                    seq = seq[200 - int(params["size"]/2):200 + int(params["size"]/2)]
                    
    
                if params['ss_type'] == "donor":
                    seq = seq[:-3]
                    seq = seq[200 - int(params["size"]/2):200 + int(params["size"]/2)]

                encoding_seq = tb.one_hot_encoding(seq)
    
                es = [int(i) for i in encoding_seq] 
                es = np.array(es, dtype=int)
    
                all_converted_sequences.append(es)
    
            # Data reshaping
            all_converted_sequences = np.array(all_converted_sequences)
            seq = all_converted_sequences.reshape(-1, params["size"], 4)
        elif params['soft'] == "splicefinder":
            labels = tf.keras.utils.to_categorical(labels)
            print(f"[INFO] Data encoding for SpliceFinder")
            for i,seq in enumerate(sequences):
                # Acceptor
                if i > 0 and i <= 355:
                    seq = seq[3:] + "N"
                # Donor
                elif i > 355 and i <= 704:
                    seq = seq[:-2]
                # Negatives
                else:
                    seq = seq.replace("N", "")

                    if i > 1050:
                        seq = seq[2:] + "NN"
                    else:
                        pass

                encoding_seq = tb.one_hot_encoding(seq)
    
                es = [int(i) for i in encoding_seq] 
                es = np.array(es, dtype=int)
                all_converted_sequences.append(es)

            seq = np.array(all_converted_sequences)
            seq = seq.reshape(-1, int(params["size"]), 4)

    else:
        if params['soft'] == "spliceator":

            # Encoding data
            print(f"[INFO] Data encoding for Spliceator")
            for seq in tqdm(sequences):
                seq = seq[200 - int(params["size"]/2):200 + int(params["size"]/2)]

                encoding_seq = tb.one_hot_encoding(seq)
    
                es = [int(i) for i in encoding_seq] 
                es = np.array(es, dtype=int)
                all_converted_sequences.append(es)
    
            seq = np.array(all_converted_sequences)
            seq = seq.reshape(-1, int(params["size"]), 4)

        elif params['soft'] == "splicefinder":
            labels = tf.keras.utils.to_categorical(labels)

            params['size'] = 400
            print(f"[INFO] Data encoding for Splicefinder")
            for seq in tqdm(sequences):
                seq = seq[200 - int(params["size"]/2):200 + int(params["size"]/2)]

                encoding_seq = tb.one_hot_encoding(seq)
    
                es = [int(i) for i in encoding_seq] 
                es = np.array(es, dtype=int)
                all_converted_sequences.append(es)
    
            seq = np.array(all_converted_sequences)
            seq = seq.reshape(-1, int(params["size"]), 4)

    predictions = model.predict(seq)
    print(predictions)

    # Evaluation
    print(f"==== {params['db_id']} - {params['ss_type']} - {params['specie']} - {params['size']} ====")  
    
    if params['soft'] == "spliceator":

        with open(os.getcwd() + f"/../Results/Final_results_SA_{params['ss_type']}.csv", "a") as file_W1:
            results = binary_confusion_matrix(predictions, labels)
            now = time.localtime(time.time())
            t = time.strftime("%y/%m/%d %H:%M", now)
            file_W1.write(f"{t};Spliceator;{params['specie']};{params['ss_type']};{params['size']};{params['subset']};{params['num']};{params['db_id']};{results[1]};{results[2]};{results[3]};{results[4]};{results[5]}\n")
    
    elif params['soft'] == "splicefinder":
        matrice_confusion_one_model(predictions, labels)

def evaluate_maxentscan(params):


    if params["soft"] == "maxentscan":
        file_name = f"Maxentscan_sequences_{params['ss_type']}_400_{params['num']}.fasta"
        output = f"Final_{params['specie']}_{params['ss_type']}.out"
        path = os.getcwd() + "/../../../Logitech/MaxEntScan/"
        path_benchmark = os.getcwd() + "/../Data/Benchmarks"       

        if os.getcwd() == path:
            pass
        else:
            os.chdir(path)

        os.system(f"perl {params['ss_type']}.pl {path_benchmark}/{params['specie']}/{file_name} > {output}")

        if params['specie'] == "Protist":
            if params['ss_type'] == "donor":
                max_pos = 346                        
            if params['ss_type'] == "acceptor":
                max_pos = 357                         
        else:
            max_pos = 10000
       
        TP, FN, FP, TN = 0,0,0,0

        with open(output, "r") as file_R1:
            for i, ligne in enumerate(file_R1):
                ligne = ligne.strip().split("\t")
                seq = ligne[0]
                score = float(ligne[1])
                if i >= max_pos:
                    if score > 0 :
                        TP +=1
                    else:
                        FN +=1
                if i < max_pos:
                    if score < 0:
                        TN += 1
                    else:
                        FP += 1
    
        all_ = TP + TN + FP + FN
        accuracy = accuracy_(TP, TN, all_)
        precision = precision_(TP, FP)
        sensitivity = sensitivity_(TP, FN)  # Recall
        specificity = specificity_(TN, FP)
        print("All:",all_, "TP:",TP,"FP:",FP,"FN:",FN,"TN:",TN)
        f1_score = round((2 * ((precision*sensitivity)/(precision+sensitivity))),2) 
        print(f"==== {params['specie']} - {params['ss_type']} ====")
        print("Accuracy - Precision - Sensitivity - Specificity - F1_score")
        print(f"{accuracy};;{precision};;{sensitivity};;{specificity};;{f1_score}")
      
    else:
        print('[INFO] Not MaxEntScan, please change the soft name and try again')
        exit()

    path = os.getcwd() + "/../../Part2_IA/Spliceator/src"
    os.chdir(path)

def evaluate_dssp(params, analyse_result=False):
    if params["soft"] == "DSSP" or params["soft"] == "dssp":
        sequences = f"DSSP_sequences_{params['ss_type']}_400_{params['num']}.fasta"
        labels = f"DSSP_labels_{params['ss_type']}_400_{params['num']}.fasta"
        output = f"Final_{params['specie']}_{params['ss_type']}.out"
        path = os.getcwd() + "/../../../Logitech/DSSP/"
        path_benchmark = os.getcwd() + "/../Data/Benchmarks"   
        if os.getcwd() == path:
            pass
        else:
            os.chdir(path)

        if params['ss_type'] == "donor":
            command_line = f"python DS_DSSP.py -I {path_benchmark}/{params['specie']}/{sequences} -O {output}"
            #os.system(f"python DS_DSSP.py -I ./Benchmarks/{params['specie']}/{sequences} -O {output}")
        elif params['ss_type'] == "acceptor":
            #os.system(f"python AS_DSSP.py -I ./Benchmarks/{params['specie']}/{sequences} -O {output}")
            command_line = f"python AS_DSSP.py -I {path_benchmark}/{params['specie']}/{sequences} -O {output}"

        print("[INFO] Ligne de commande a coller dans le dossier DSSP\n")
        print(command_line + "\n")

        if analyse_result:
                           
            if params['specie'] == "Protist":
                if params['ss_type'] == "donor":
                    max_pos = 346                         
                if params['ss_type'] == "acceptor":
                    max_pos = 357                        
            else:
                max_pos = 10000
           
            TP, FN, FP, TN = 0,0,0,0
            
            with open(os.getcwd()+"/" + output, "r") as file_R1:
                for i,ligne in tqdm(enumerate(file_R1)):
                    ligne = ligne.strip().split(";")
                    id_score = ligne[0]
                    score = float(ligne[1])
                    
                    # Neg
                    if i < max_pos:
                        if score < 0.5:
                            TN += 1
                        else:
                            FP += 1
                    # pos
                    else:
                        if score >= 0.5:
                            TP += 1
                        else:
                            FN += 1
            print(TP, TN, FP, FN)
            all_ = TP + TN + FP + FN
            accuracy = accuracy_(TP, TN, all_)
            precision = precision_(TP, FP)
            sensitivity = sensitivity_(TP, FN)  # Recall
            specificity = specificity_(TN, FP)
            print("All:",all_, "TP:",TP,"FP:",FP,"FN:",FN,"TN:",TN)
            f1_score =round((2 * ((precision*sensitivity)/(precision+sensitivity))),2) 
            print(f"==== {params['specie']} - {params['ss_type']} ====")
            print("Accuracy - Precision - Sensitivity - Specificity - F1_score")
            print(f"{accuracy};;{precision};;{sensitivity};;{specificity};;{f1_score}")
      
    else:
        print('[INFO] Not DSSP, please change the soft name and try again')
        exit()

    path = os.getcwd() + "/../../Part2_IA/Spliceator/src"
    os.chdir(path)

def evaluate_genesplicer(params, analyse_result=False):
    if params["soft"] == "genesplicer":
        file_name = f"GeneSplicer_sequences_{params['ss_type']}_400_{params['num']}.fasta"
        output = f"Final_{params['specie']}_{params['ss_type']}.out"
        path = os.getcwd() + "/../../../Logitech/GeneSplicer/sources/"
        path_benchmark = os.getcwd() + "/../Data/Benchmarks"       

        if os.getcwd() == path:
            pass
        else:
            os.chdir(path)
        #Human, Drosophila, arabidopsis, 
        if params['specie'] == 'Human':
            spe = "human"
        elif params['specie'] == "Plant":
            spe = "arabidopsis"
        elif params['specie'] == "Fly":
            spe = "drosophila"
        elif params['specie'] == "Protist":
            spe = "pyoelii"
        else:
            print("[INFO] Model doesn't exist")
            exit()

        #print(f"genesplicer {path_benchmark}/{file_name} ../{spe}/ > {output}")
        #os.system(f"genesplicer {path_benchmark}/{file_name}/ ../{spe}/ > {output}")

        print("[INFO] à coller dans le dossier Genesplicer\n")
        print(f"\ngenesplicer ../../../Part2_IA/Spliceator/Data/Benchmarks/{params['specie']}/{file_name}/ ../{spe} > {output}\n")

        
        if params['specie'] == "Protist":
            if params['ss_type'] == "donor":
                max_pos = 346     
                maxi = 707                     
            if params['ss_type'] == "acceptor":
                max_pos = 357   
                maxi = 707                      
        else:
            nbr_pos = 10000
            maxi = 20000        

        if analyse_result:
            TN,TP,FP,FN = 0,0,0,0
    
            liste = []
            # de 0 Ã  X = negatif et de X+1 Ã  max = positif
            #On genere la liste des i positifs  
            for i in range(nbr_pos,maxi):
                liste.append(i)
    
            # ouverture du fichier des rÃ©sultats
            with open(output, "r") as file_R1:
                for i, ligne in enumerate(file_R1):
                    if i != 0 :
                        ligne = ligne.strip().split(" ")
            
                        ss_type2 = ligne[4]
                        pos = int(ligne[0]) +1
            
                        if ss_type2 == params['ss_type']:
            
                            res = pos
                            if res in liste:
                                TP+=1
                            else:
                                FP+=1
            
                        FN = nbr_pos - TP
                        TN = nbr_pos - FP
        
            all_ = TP + TN + FP + FN
            accuracy    = accuracy_(TP, TN, all_)
            precision   = precision_(TP, FP)
            sensitivity = sensitivity_(TP, FN)  # Recall
            specificity = specificity_(TN, FP)
            print("All:",all_, "TP:",TP,"FP:",FP,"FN:",FN,"TN:",TN)
            f1_score = round((2 * ((precision*sensitivity)/(precision+sensitivity))),2) # si faible, soit bcp de FP ou FN
            print(f"==== {params['specie']} - {params['ss_type']} ====")
            print("Accuracy - Precision - Sensitivity - Specificity - F1_score")
            print(f"{accuracy};;{precision};;{sensitivity};;{specificity};;{f1_score}")
          
    else:
        print('[INFO] Not Genesplicer, please change the soft name and try again')
        exit()

    path = os.getcwd() + "/../../../Part2_IA/Spliceator/src"
    os.chdir(path)

def show_result():
    df = create_df()
    print(df)
    liste_specie = df.Specie.unique().tolist()
    liste_db = df.Db_id.unique().tolist()
    liste_size = df.Size.unique().tolist()
    
    # Display result for tab S1 or S2
    for db_id in liste_db:
        for specie in liste_specie:
            txt = []
            for size in liste_size:
                ligne = df.loc[(df['Db_id'] == db_id) & (df['Specie'] == "Human") & (df['Size'] == size)]['Accuracy']
                print(db_id, specie, size, ligne)
                txt.append(ligne)

def main():

    liste_spe = ["Human", "Danio", "Fly", "Worm", "Thaliana", "Protist"]

    for specie in liste_spe:
        params = {"size":       200,
                  "subset":     "GS_1",
                  "db_id":      "db_3",
                  "ss_type":    "donor",
                  "num":        "Final_3",
                  "specie":     specie,
                  "model_name": "",
                  "soft": "spliceator"}

        evaluate_spliceator_or_splicefinder(params)
        
    

if __name__ == '__main__':
    main()