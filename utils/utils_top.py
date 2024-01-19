#%% 
import random
import numpy as np
import sys
import csv

sys.path.append("C:/Users/lucas/Documents/GitHub/RNN_gradient_problem/raw_data")
main_path = "C:/Users/lucas/Documents/GitHub/RNN_gradient_problem/"

def generate_sequences(nb_samples, T=50):
    sequences = list()
    labels = list()
    distractors = ['c', 'd', 'e', 'f']
    symbols = ['A', 'B']

    for _ in range(nb_samples):
        pos1 = random.randint(T//10, 2*T//10)
        pos2 = random.randint(4*T//10, 5*T//10)
        label = []

        sequence = np.random.choice(distractors, T)
        label = np.random.choice(symbols, 2)
        sequence[pos1] = label[0]
        sequence[pos2] = label[1]

        sequences.append(''.join(sequence))
        labels.append(''.join(label))

    return sequences, labels

def save_sequences(sequences, labels, filename):
    data = np.column_stack((sequences, labels))
    column_names = ['sequence', 'label']
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
        writer.writerows(data)

def load_sequences(path, T=50):
    ''' path (str) : path to a folder containing three files train_Tx.csv, test_Tx.csv and eval_Tx.csv where x is the value of T
        return six lists : X_train, y_train, X_eval, X_test, y_eval, y_test
    '''
    X_train, y_train = list(), list()
    X_eval, y_eval = list(), list()
    X_test, y_test = list(), list()

    with open(path + "/train_T" + str(T) + ".csv", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sequence = list(row['sequence'])
            label = list(row['label'])
            X_train.append(sequence)
            y_train.append(label)

    with open(path + "/eval_T" + str(T) + ".csv", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sequence = list(row['sequence'])
            label = list(row['label'])
            X_eval.append(sequence)
            y_eval.append(label)

    with open(path + "/test_T" + str(T) + ".csv", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sequence = list(row['sequence'])
            label = list(row['label'])
            X_test.append(sequence)
            y_test.append(label)
    
    return X_train, y_train, X_eval, X_test, y_eval, y_test


def letters_to_onehot(L):
    L_number = list(map(lambda x: 0 if x=='A' else (1 if x=='B' else ord(x)-97), L))
    return np.eye(6, dtype='float')[L_number]


def list_to_tensor(X_list):
    ''' X_list: a list of lists of chars. For X only'''
    # return torch.Tensor(np.array([letters_to_onehot(L) for L in X_list]))
    return np.array([letters_to_onehot(L) for L in X_list])


def label_to_tensor(y_list):
    # return torch.Tensor(np.array([[label[0]=='A', label[1]=='A'] for label in y_list ], dtype='uint8'))
    return np.array([[label[0]=='A', label[1]=='A'] for label in y_list ], dtype='float')


def load_tensors(path, T):
    X_train, y_train, X_eval, X_test, y_eval, y_test = load_sequences(path, T=T)

    X_train, y_train = list_to_tensor(X_train), label_to_tensor(y_train)
    X_eval, y_eval = list_to_tensor(X_eval), label_to_tensor(y_eval)
    X_test, y_test = list_to_tensor(X_test), label_to_tensor(y_test)

    return X_train, y_train, X_eval, X_test, y_eval, y_test

train_nb_samples = 50000
test_eval_nb_sample = 20000

# for T in [50, 100, 150, 200]:
#     X_train, y_train = generate_sequences(train_nb_samples, T)
#     save_sequences(X_train, y_train, main_path + "raw_data/top/train_T" + str(T) + ".csv")


# for T in range(50, 401, 50):
#     X_test, y_test = generate_sequences(test_eval_nb_sample, T)
#     X_eval, X_test, y_eval, y_test = train_test_split(X_test, y_test, test_size=0.5)
#     save_sequences(X_eval, y_eval, main_path + "raw_data/top/eval_T" + str(T) + ".csv")
#     save_sequences(X_test, y_test, main_path + "raw_data/top/test_T" + str(T) + ".csv")


# X_train, y_train, X_eval, X_test, y_eval, y_test = load_tensors(main_path + "raw_data/top", T=50)
