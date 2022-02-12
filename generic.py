from typing import Literal, Union


def read_intents_file(filepath: str) -> dict:
    """
    Opens and reads data from a json file.

    Parameters
    -----------
    filepath (str): Path to the json file.

    Returns
    --------
        data (dict): Data read from the json and parsed as a Python dictionary.
    """
    import json
    
    with open(filepath) as f:
        data = json.loads(f.read())

    return data


def read_database_file(filepath: str) -> list[dict[Literal["disease"]:str, Literal["symptoms"]:list[str]]]:
    """
    Opens and reads data from a database file.

    Parameters
    -----------
    filepath (str): Path to the database file.

    Returns
    --------
        list[
            dict[
                Literal["disease"]: (str): Disease name,
                Literal["symptoms"]: (list[str]): List of symptoms.
            ]
        ]
    """
    import sqlite3
    
    with sqlite3.connect(filepath) as conn:
        cur = conn.cursor()
        command = """SELECT
        diseases.disease_name,
        symptoms.symptom_description symptom_description
        FROM diseases JOIN symptoms on diseases.symptom_description = symptoms.symptom_id"""
        temp =[]
        
        dict_ = {}
        
        for disease, symptom in cur.execute(command):
            if disease in dict_:
                dict_[disease].append(symptom)
            else:
                dict_[disease] = [symptom]
        for key, value in dict_.items():
            temp.append(
                {
                    "disease": key,
                    "symptoms": value
                }
            )
        
    return temp


def stem_words(words: list[str]) -> list[str]:
    """
    Replaces all the words in `words` and replaces them with their roots.
    Example: "what's"->"what", "there?"->"there", "goodbye!"->"goodbye", etc.

    Parameters
    -----------
        words (list[str]): List of tokenized words.

    Returns
    --------
        words (list[str]): List of stemmed words.
    """
    from nltk.stem.lancaster import LancasterStemmer

    stemmer = LancasterStemmer()
    words = [stemmer.stem(w.lower()) for w in words]

    return words


def read_name(text: str, nlp) -> Union[list[tuple[str, str]], list[None]]:
    """
    Checks string for a name.
    
    Parameters
    -----------
        text (str): String to check for a name in.

    Returns
    --------
        Union[
            list[tuple[str, str]]
            list[None]
        ]: Either a list with a tuple that contains two values, person's name and entity name; example: `("Bob", "PERSON")`, or an empty list if no name was found.
    """

    doc = nlp(text.strip())

    return [(X.text, X.label_) for X in doc.ents if X.label_ == "PERSON"]
