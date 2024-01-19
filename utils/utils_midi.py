#%% 
import pretty_midi
import IPython
import numpy as np
import matplotlib.pyplot as plt
import glob
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame


## Constants
n_x = 128  # the number of notes
max_midi_T_x = 1000  # the maximum number of notes we read in each midi_file
model_T_x = 100  # the length of the sequences considered for the RNN model


def to_one_hot_vector(x, nb_classes):
    return np.eye(nb_classes, dtype='uint8')[x]


def convert_midi_to_list(midi_file_list, max_midi_T_x=max_midi_T_x):
    """
    read the notes within all midi files
    truncate the length if > max_midi_T_x

    Parameters: 
    midi_file_l: list of MIDI files
    max_midi_T_x: the maximum number of notes we read in a given midi_file

    Returns X_list: a list of np.array X_ohe of size (midi_T_x, n_x) which contains the one-hot-encoding representation of notes over time
    """
    X_list = []
    for midi_file in midi_file_list:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        note_list = [note.pitch for note in midi_data.instruments[0].notes]
        midi_T_x = len(note_list) if len(note_list) < max_midi_T_x else max_midi_T_x
        X_ohe = to_one_hot_vector(note_list[:midi_T_x], nb_classes=n_x)
        X_list.append(X_ohe)
    return X_list


def save_npy_files(output_path, midi_file_list, max_midi_T_x=max_midi_T_x):
    """
    read the notes within all midi files
    truncate the length if > max_midi_T_x
    save the arrays in npy files

    Parameters: 
    output_path: string with path to the folder where npy files are saved
    midi_file_l: list of MIDI files
    max_midi_T_x: the maximum number of notes we read in a given midi_file
    """
    for midi_file in midi_file_list:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
        except:
            pass
        note_list = [note.pitch for note in midi_data.instruments[0].notes]
        midi_T_x = len(note_list) if len(note_list) < max_midi_T_x else max_midi_T_x
        X_ohe = to_one_hot_vector(note_list[:midi_T_x], nb_classes=n_x)
        name = midi_file.split("/")[-1].split("\\")[-1].split(".")[0] + ".npy"
        np.save(output_path + "/" + name, X_ohe)


def load_npy_files(folder_path):
    """ Returns X_list: a list of np.array X_ohe of size (midi_T_x, n_x) which contains the one-hot-encoding representation of notes over time
    """
    return [ np.load(file) for file in glob.glob(folder_path + "/*.npy") ]


def display_notes(notes_list=None, X_ohe=None, midi_file=None):
    if notes_list is not None:
        # Conversion in ohe
        midi_T_x = len(notes_list) if len(notes_list) < max_midi_T_x else max_midi_T_x
        ohe = to_one_hot_vector(notes_list[midi_T_x-1], nb_classes=n_x)
        plt.figure(figsize=(10, 4))
        plt.imshow(ohe.T, aspect='auto')
        plt.set_cmap('gray_r')
        plt.grid(True)
    
    elif X_ohe is not None:
        plt.figure(figsize=(10, 4))
        plt.imshow(X_ohe.T, aspect='auto')
        plt.set_cmap('gray_r')
        plt.grid(True)

    elif midi_file is not None:
        # conversion in ohe
        ohe = convert_midi_to_list([midi_file])[0]
        plt.figure(figsize=(10, 4))
        plt.imshow(ohe.T, aspect='auto')
        plt.set_cmap('gray_r')
        plt.grid(True)
    
    else:
        raise RuntimeError("nothing to display")
    

def convert_list_to_data(X_list, model_T_x=model_T_x, sequence_step=1):
    """
    convert X_list to input X_train and output Y_train training data

    X_list: a list of np.array X_ohe of size (midi_T_x, n_x) which contains the one-hot-encoding representation of notes over time

    Returns 
    X_train: the set of all m input sequences; np.array of shape (m, model_T_x, n_x)
    Y_train: the set of all m output sequences; np.array of shape (m, model_T_x, n_x)

    note: m is the total number of training items, it is be larger than the number of MIDI files since we use several starting time t in each MIDI file
    """
    X_train_list = []
    Y_train_list = []
    for i in range(min(80,len(X_list))):
        for t in range(len(X_list[i])-model_T_x):
            X_item = X_list[i][t:t+model_T_x]
            Y_item = X_list[i][t+sequence_step:t+model_T_x+1]
            X_train_list.append(X_item)
            Y_train_list.append(Y_item)
    X_train = np.asarray(X_train_list)
    Y_train = np.asarray(Y_train_list)

    return X_train, Y_train


def get_max_temperature(proba_v, temperature=1):
    """
    apply a temperature to the input probability
    consider it as a multinomial distribution
    sample it

    proba_v: np.array(n_x) input probability vector
    temperature: scalar float

    Returns
    index_pred: (int) position of the sampled data in the probability vector
    pred_v: np.array(n_x) modified probability
    """
    exp_v = np.exp(np.log(proba_v)/temperature)
    pred_v = exp_v / np.sum(exp_v)
    rng = np.random.default_rng()
    # index_pred = np.argmax(pred_v)
    index_pred = np.argmax(rng.multinomial(1, pred_v))  ## Sometimes raise ValueError: sum(pvals[:-1]) > 1.0    

    return index_pred, pred_v


def sample_new_sequence(model, prior_v):
    """
    sample the trained language model to generate new data

    model: trained language model

    Returns
    note_list: (list of int) list of generated notes (list of their index)
    prediction_list: (list of np.array(n_x)) list of prediction probabilies over time t (each entry of the list is one of the y[0,t,:])
    """

    prediction_list = []  
    pred_v = prior_v
    note_list = []  
    
    start_note_index = np.random.choice(n_x, p=prior_v)
    note_list.append(start_note_index)

    # Initializing x
    x = np.zeros((1,model_T_x, n_x))
    x[0,0,start_note_index] = 1

    # Loop to generate new notes
    for t in range(1,model_T_x):
        next_note_probabilities = model(x)   ### CHANGE HERE FOR MODEL
        index_pred, pred_v = get_max_temperature(next_note_probabilities[0,t-1,:])
        note_list.append(index_pred)
        prediction_list.append(pred_v)
        x[0,t,index_pred] = 1

    return note_list, prediction_list


def onehot_to_values(x):
    return np.argmax(x, axis=-1)


def convert_list_to_midi(notes_list, output_path='output.mid'):
    new_midi_data = pretty_midi.PrettyMIDI()
    cello_program = pretty_midi.instrument_name_to_program('Cello')
    cello = pretty_midi.Instrument(program=cello_program)
    time = 0
    step = 0.3
    for note_number in notes_list:
        myNote = pretty_midi.Note(velocity=100, pitch=note_number, start=time, end=time+step)
        cello.notes.append(myNote)
        time += step
    new_midi_data.instruments.append(cello)
    new_midi_data.write(output_path)
    return new_midi_data


def play_midi(midi_data=None, midi_path=None):   # does not work currently
    if midi_data:
        audio_data = midi_data.synthesize(fs=44100)
        IPython.display.Audio(audio_data, rate=44100)
    
    elif midi_path:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        audio_data = midi_data.synthesize(fs=44100)
        IPython.display.Audio(audio_data, rate=44100)
    
    else:
        raise ValueError("nothing to play")
    

def play_music(midi_filename):
    '''Stream music_file in a blocking manner'''
    pygame.mixer.init(44100,-16,2,2048)
    clock = pygame.time.Clock()
    pygame.mixer.music.load(midi_filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30) 

