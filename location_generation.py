import random
import country_converter as coco
from geopy import geocoders
import pandas as pd;
import numpy as np
from multiprocessing.context import Process
import multiprocessing
import os;
import re;
from pandas import json_normalize;
import json

import tensorflow.python.compat as tf
from tensorflow.python.keras.layers import Activation, Input, SimpleRNN, Dense, LSTM
from tensorflow.python.keras.models import Model, Sequential, model_from_json
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras import utils
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

def city_data():
    cities_csv = pd.read_csv(
        "DataCopies/cities5000.txt",
         delimiter="\t",
          #Tags for each column header
          names=["geonameid", "name", "asciiname", "alternatenames",
          "latitude", "longitude", "feature class", "feature code", "country code", "cc2", 
          "admin1 code", "admin2 code",
          "admin3 code", "admin4 code", "population", "elevation", "dem", "timezone", "modification date"])
    # print(cities_csv)
    # print(cities_csv.columns)
    # print(cities_csv["geonameid"])
    #column headers that we want to drop
    cities_csv.drop(["geonameid", "latitude", 
    "longitude", "feature class", "feature code", 
    "admin1 code", "admin2 code", "admin3 code", 
    "admin4 code", "modification date"], axis=1, inplace=True)
    return cities_csv


def create_names():
    # Grab correct city cata for dataframe
    df = city_data()
    #print(df)
    # Get unique names
    names = pd.unique(df["timezone"])
    # Retain original copy
    df_out = df.copy()
    df_new = pd.DataFrame(columns=df_out.columns)
    # Loop over unique timezones
    for name in names:
        df_temp = df_out[df_out["timezone"] == name]
        # Number of results
        df_size = int(df_temp["timezone"].size)
        # print("Size of dataframe is ", df_temp["timezone"].size)
        # print(df_temp)
        try:
            if df_size >= 500:
                df_new = pd.concat([df_new, df_temp.sample(500)], ignore_index=True)
            elif df_size >= 100:
                #print("I WORKED")
                df_new = pd.concat([df_new, df_temp.sample(100)], ignore_index=True)
            elif df_size >= 50:
                df_new = pd.concat([df_new, df_temp.sample(50)], ignore_index=True)
            elif df_size == 0:
                df_new = df_temp.sample(500)
        except:
            print("Error")
    # print(df_new)
    # print(df_new["timezone"].size)
    df_new = df_new.drop_duplicates()
    # print(df_new.size)
    # print(pd.unique(df_new["timezone"]))
    #print(input(""))
    print(df.columns)
    # Converting country code with coco, get country codes
    tags = pd.unique(df["country code"])
    # print(tags)
    # print(df["country code"])
    # Dict init
    country_names = {}
    for tag in tags:
        # print(tag)
        # One of the tags is assigned as "nan" for some reason? unsure why, i will assume that it is an issue with the data set rather than the data processing
        # print(f"Tag: {tag}, Type: {type(tag)} ")
        if isinstance(tag, str):
            # Add values to country_names key using converted country tag
            country_names[tag] = coco.convert(names=tag, to="name_short")
    #print(country_names)
    # gn = geocoders

    names = pd.unique(df["timezone"])
    print(list(pd.unique(df["timezone"])))
    # New data structure we will push to
    df_fantasy = pd.DataFrame(columns=["name", "result", "capital", "country"])
    #print(df["timezone"].value_counts())
    for f in names:
        df_temp = df_new[df_new["timezone"] == f]
        for k, v in df_temp.iterrows():
            #print(k, v)
            # Get name of capital city using split
            timezone = v["timezone"].split(r"/")[1]

            df_fantasy = df_fantasy.append({"name": v["name"], "result": v["asciiname"], "capital": timezone, "country": country_names.get(v["country code"])}, ignore_index=True)
            #print("DF fantasy ", df_fantasy)
        #print(df_temp)
    # print(df_fantasy)
    # print("LOOK HERE ", df_fantasy["capital"].value_counts())
    #print(input(""))
    df = df_fantasy
    # print(df)
    # print(df.size)
    #print(input(""))



    # https://github.com/JKH4/name-generator using this as a basis for the name generator, thanks to JKH4
    # https://towardsdatascience.com/generating-pok%C3%A9mon-names-using-rnns-f41003143333
    print("Attempting to create new names using previous names")
    #Tagging names with abnormal values
    padd_start, padd_end = "#", "*"
    df["result"] = df["result"].map(lambda n: str(padd_start) + str(n) + str(padd_end))
    # df.describe()
    # print('Example of names to be cleaned:')
    # df = df.loc[~(df["option"] == "Clan")]
    # df = df.loc[~(df["option"] == "Virtue")]
    # df = df.loc[~(df["option"] == "Duergar Clan")]
    # df = df.loc[~(df["option"] == "Family")]
    # option_names = ['Female', 'Male', 'Child', 'Female Adult', 'Male Adult']
    # for i in option_names:
    #     df = df.loc[~(df["option"] == i)]

    # "Virtue", "Duergar Clan", "Family"])]

    #print('Max name size: {}'.format(df['name'].map(len).max()))
    # print("--\n")

    # Get identifiers again
    nationality = list(pd.unique(df["country"]))
    origins = list(pd.unique(df["capital"]))
    print(list(pd.unique(df.keys())))
    print(list(pd.unique(df["timezone"])))
    print(origins)
    ini = input("HERE!")
    data_dict = {}
    for r in nationality:
        try:
            data_dict[r] = {}
            data_dict[r]["country"] = r
            data_dict[r]["name_list"] = df[df["country"] == r]["result"]
            data_dict[r]["char_list"] = sorted(list(set(data_dict[r]["name_list"].str.cat() + "*")))
            data_dict[r]["char_to_num"] = {ch: i for i, ch in enumerate(data_dict[r]["char_list"])}
            data_dict[r]["ix_to_char"] = {i: ch for i, ch in enumerate(data_dict[r]["char_list"])}

            # for k, v in data_dict.items():
            #     print('group: {}'.format(k))
            #     print('  - number of names: {} ({}, ...)'.format(len(v['name_list']), v['name_list'][:5].tolist()))
            #     print('  - number of chars: {}'.format(len(v['char_list'])))
            #     print('  - chars: {}'.format(v['char_list']))
            #     print('  - char_to_num: {}'.format(v['char_to_num']))
            #     print('  - ix_to_char: {}'.format(v['ix_to_char']))
            #     print('######################')
        except:
            pass
    names_dict = {}
    print("Data dict = ", data_dict, " Type =", type(data_dict))
    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    # Process list
    procs = []
    for g in nationality:
        # print(g)
        # Create new process using function generate_new_names, args in tuple form
        proc = Process(target= generate_new_names, args= (data_dict, names_dict, g),)
        procs.append(proc)
        proc.start()
    for p in procs:
        p.join()
    print("Did that work?")
    return names_dict

def generate_new_names(data_dict, names_dict, g):
    x, y, train_util, train_info = training_data(g, data_dict, 3)
        # print(train_util)
    current_model, training_infos, history = model_start(train_info, 172)  # Original used 128, 256 is too slow
    compile_model(model=current_model,
                      hyperparams={"lr": 0.003, "loss": "categorical_crossentropy", "batch_size": 32},
                      history=history, training_infos=training_infos)
    train_model(current_model, x, y, training_infos, history, 950)  # Epochs after 2000 seem efficient
    print("Printing {} names".format(g))
    name_list = []
    name_list = set(name_list)
    i = 0
    vowels = "aeiou"
    while i < 950:
        name = generate_name(
                model=current_model,
                trainset_infos=train_info,
                #         sequence_length=trainset_infos['length_of_sequence'],
                train_util=train_util,
                #         padding_start=padding_start,
                #         padding_end=padding_end,
                name_max_length=15)
        if len(name) >= 3 and int(name.lower().count("z")) < 4:
            vow_check = [vow for vow in name.lower() if vow in vowels]
            if len(vow_check) >= 1:
                name_list.add(name.title())
        i += 1
    # print(i)
    # print(len(name_list))
    name_list = sorted(name_list)
    names_dict.update({g: list(name_list)})
    # print(names_dict)
    out_df = pd.DataFrame()
    out_df["Name"] = name_list
    out_df["Origin"] = g
    out_df.to_csv(path_or_buf="AI_OUTPUT/{}.csv".format(g))


def training_data(target_group, data_dict, len_sequence):
    # print(target_group)
    # print(data_dict)
    train_names = data_dict[target_group]["name_list"].tolist()
    padd_start, padd_end = train_names[0][0], train_names[0][
        -1]  # First element of list, with first and last character id'd
    char_to_index = data_dict[target_group]["char_to_num"]
    index_to_char = data_dict[target_group]["ix_to_char"]

    num_chars = len(data_dict[target_group]["char_list"])
    trainable_names = [padd_start * (len_sequence - 1) + n + padd_end * (len_sequence - 1) for n in train_names]
    x_list, y_list = [], []
    for name in train_names:
        for i in range(max(1, len(name) - len_sequence)):
            new_seq = name[i:i + len_sequence]
            target_char = name[i + len_sequence]

            x_list.append([utils.to_categorical(char_to_index[c], num_chars) for c in new_seq])
            y_list.append(utils.to_categorical(char_to_index[target_char], num_chars))
    x = np.array(x_list)
    y = np.array(y_list)

    m = len(x)
    trainset_infos = {
        'target_group': target_group,
        'length_of_sequence': len_sequence,
        'number_of_chars': num_chars,
        'm': m,
        'padding_start': padd_start,
        'padding_end': padd_end,
    }

    out_dict = {"c2i": char_to_index, "i2c": index_to_char}
    return x, y, out_dict, trainset_infos


def model_start(trainset_infos, lstm_units):
    len_seq = trainset_infos["length_of_sequence"]
    num_char = trainset_infos["number_of_chars"]

    x_in = Input(shape=(len_seq, num_char))

    x = LSTM(units=lstm_units)(x_in)  # Default 256
    x = Dense(units=num_char)(x)

    output = Activation("softmax")(x)

    model = Model(inputs=x_in, outputs=output)
    training_infos = {
        'total_epochs': 0,
        'loss': 0,
        'acc': 0,
        'trainset_infos': trainset_infos,
    }
    history = {
        'loss': np.array([]),
        'accuracy': np.array([]),
        'hyperparams': []
    }
    model.summary()
    return model, training_infos, history


def compile_model(model, hyperparams, history, training_infos):
    optimizer = Adam(learning_rate=hyperparams["lr"])
    model.compile(loss=hyperparams["loss"], optimizer=optimizer, metrics=["accuracy"])
    history["hyperparams"].append((training_infos["total_epochs"], hyperparams))
    # print("\n\n\n", "History of params", history["hyperparams"])

    return None


def train_model(model, x, y, training_infos, history, epochs_to_add=100):
    # history["acc"] = history.pop("accuracy")
    # history["hyperparams"] = history.pop("hyperparams")
    old_loss = training_infos['loss']
    old_acc = training_infos['acc']
    # Extract hyperparams to fit the model
    hyperparams = history['hyperparams'][-1][1]

    # Train the model
    training_model = model.fit(
        x, y,
        batch_size=hyperparams['batch_size'],
        initial_epoch=training_infos['total_epochs'],
        epochs=training_infos['total_epochs'] + epochs_to_add
    )

    # Update history
    for key, val in training_model.history.items():
        history[key] = np.append(history[key], val)

    # Update the training session info
    training_infos['total_epochs'] += epochs_to_add
    training_infos['loss'] = history['loss'][-1]
    training_infos['acc'] = history['accuracy'][-1]

    return None


def generate_name(
        model, trainset_infos, train_util,
        name_max_length=25
):
    dict_size = trainset_infos["number_of_chars"]
    seq_len = trainset_infos["length_of_sequence"]
    index_to_char = train_util["i2c"]
    char_to_index = train_util["c2i"]
    padd_start = trainset_infos["padding_start"]
    generated_name = padd_start * (seq_len + name_max_length)
    probability = 1
    gap = 0
    for i in range(name_max_length):
        x_char = generated_name[i:i + seq_len]
        x_cat = np.array([[utils.to_categorical(char_to_index[c], dict_size) for c in x_char]])
        p = model.predict(x_cat)
        best_char, best_char_prob = index_to_char[np.argmax(p)], np.max(p)

        new_char_index = np.random.choice(range(dict_size), p=p.ravel())
        new_char_prob = p[0][new_char_index]

        new_char = index_to_char[new_char_index]
        generated_name = generated_name[:seq_len + i] + new_char + generated_name[seq_len + i + 1:]
        probability *= new_char_prob
        gap += best_char_prob - new_char_prob
        # print(
        #     'i={} new_char: {} ({:.3f}) [best:  {} ({:.3f}), diff: {:.3f}, prob: {:.3f}, gap: {:.3f}]'.format(
        #         i, new_char,new_char_prob,
        #         best_char,best_char_prob,
        #         best_char_prob - new_char_prob,
        #         probability,gap
        #     ))
        if new_char == trainset_infos['padding_end']:
            break
    generated_name = generated_name.strip("#*")
    # print(generated_name.title())
    # print('{} (probs: {:.6f}, gap: {:.6f})'.format(generated_name, probability, gap))
    return generated_name  # , {'probability': probability, 'gap': gap}

if __name__ == "__main__":
    create_names()