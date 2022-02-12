import gc
import sys
import pickle
import random
# from typing import


import torch


import generic
import train_on_intents
import train_on_database
from User import User


def welcome_message() -> None:
    """
    Function to print a welcome message when the bot is first ran.
    """
    print("Please keep in mind that this is by no means a proper diagnosis,")
    print("treat the result you get from this bot as mere probabilities")
    print("and nothing more.")
    print("Now begins your conversation with Medbot!")
    print("The conversational A.I. that will diagnose your illness, provided symptoms.")
    print("Do start by greeting the bot!")
    print("--------------------------------------------\n")


def load_intents_model(
    *,
    pickle_filepath: str,
    model_filepath: str,
) -> tuple[list[str], list[str], torch.nn.Module]:
    """
    Loads list of all pattern words and tags in the intents.json file, and the trained model from pickle file.

    Parameters
    -----------
        pickle_filepath (str): Path to saved pickle data.
        model_filepath (str): Path to saved model.

    Returns
    --------
        tuple[
            words (list[str]): List of all pattern words in intents.json, loaded from pickle file.
            tags (list[str]): List of all tags in intents.json, loaded from pickle file.
            model (torch.nn.Module): Trained model.
        ]
    """
    with open(pickle_filepath, "rb") as f:
        words, tags, training, output = pickle.load(f)
    model = train_on_intents.train_and_save_or_load_model(
        training, output, filepath=model_filepath, train=False
    )
    return words, tags, model


def load_database_model(
    *,
    pickle_filepath: str,
    model_filepath: str,
) -> tuple[list[str], list[str], torch.nn.Module]:
    """
    Loads list of all pattern words and tags in the intents.json file, and the trained model from pickle file.

    Parameters
    -----------
        pickle_filepath (str): Path to saved pickle data.
        model_filepath (str): Path to saved model.

    Returns
    --------
        tuple[
            words (list[str]): List of all pattern words in intents.json, loaded from pickle file.
            tags (list[str]): List of all tags in intents.json, loaded from pickle file.
            model (torch.nn.Module): Trained model.
        ]
    """
    with open(pickle_filepath, "rb") as f:
        words, tags, training, output = pickle.load(f)
    model = train_on_database.train_and_save_or_load_model(
        training, output, filepath=model_filepath, train=False
    )
    return words, tags, model


def bag_of_words_from_input(inp: str, words: list[str]) -> torch.Tensor:
    """
    Creates a list of zeroes and ones based on whether words from input are in the total list of words or not.

    Parameters
    -----------
        inp (str): User input.
        words (list[str]): List of words.

    Returns
    --------
        torch.Tensor: Returns a Rank 1 tensor of zeroes and ones based on whether words from input are in the total list of words or not.
    """
    import nltk

    bag = [0] * len(words)
    tokenized_inp = nltk.word_tokenize(inp)
    stemmed_inp = generic.stem_words(tokenized_inp)
    bag = [1 if word in stemmed_inp else 0 for word in words]

    return torch.tensor(bag, dtype=torch.float)


def get_symptoms(handle: str) -> User:
    """
    Gets symptoms from user.
    
    Parameters
    -----------
        handle (str): The user handle for the user.

    Returns
    --------
        user (User): A dataclass storing all data from user.
    """
    user = User(handle)
    
    print(f"<Medbot>: Hi, {user.first_name}.")
    print("<Medbot>: Can you tell me of the symptoms you're having? (quit to stop)", end='\n\n')
    
    confirmations = ["Is that all?", "Are you done?", "Move on to the diagnosis?"]
    
    inp = input(f"<{handle}>: ")
    print()
    while inp.lower() != 'quit':
        symptoms = ' '.join(map(str.strip, inp.split(',')))
        user.symptoms += f' {symptoms}'
        print(f"<Medbot>: {random.choice(confirmations)}", end='\n\n')
        inp = input(f"<{handle}>: ")
        print()
        
    return user
        

def chat():
    """
    Interface function to hold conversation with the bot.
    """
    
    import spacy
    nlp = spacy.load("en_core_web_trf")
    
    welcome_message()
    
    handle = "You"
    
    intents_data = generic.read_intents_file("json_file/intents.json")
    words, tags, intents_model = load_intents_model(
        pickle_filepath="pickle/intents.pickle", model_filepath="model/intents_model.pt"
    )
    
    while True:
        inp = input(f"<{handle}>: ")
        print()
        if inp.lower() == "quit":
            break

        if handle == "You":
            if (username := generic.read_name(inp, nlp)) != []:
                try:
                    del spacy
                    del nlp
                    del words
                    del tags
                    del intents_model
                    del intents_data
                    del inp
                    del results
                    gc.collect()
                except NameError:
                    pass
    
                symptoms, diseases, database_model = load_database_model(
                    pickle_filepath="pickle/database.pickle", model_filepath="model/database_model.pt"
                )
                
                user = get_symptoms(username[0][0].split()[0].capitalize())
                inp = bag_of_words_from_input(user.symptoms, words=symptoms)
                database_results = database_model.predict(inp.view(-1, inp.shape[0]))
                print("Probabilities:")
                for disease, result in zip(diseases, *database_results.tolist()):
                    print(f"{disease}:\t\t{round(result*100, 2)} %")
                
                print(f"\nHighest probability of: {diseases[torch.argmax(database_results)]}")
                
                sys.exit(f"\nTake care!")

        inp = bag_of_words_from_input(inp, words=words)
        results = intents_model.predict(inp.view(-1, inp.shape[0]))
        result_index = torch.argmax(results)
        tag = tags[result_index]

        for intent in intents_data["intents"]:
            if intent["tag"] == tag:
                responses = intent["responses"]
                break

        print(f"<Medbot>: {random.choice(responses)}")
        print()


if __name__ == "__main__":
    chat()
