from __future__ import annotations

import gc
import sys
import pickle
import random
from typing import Literal
# from typing import


import nltk
import torch
import spacy
nlp = spacy.load("en_core_web_trf")
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc


import generic
import train_on_intents
import train_on_database
from User import User


import sys


def get_welcome_message() -> None:
    """
    Function to print a welcome message when the bot is first ran.
    """
    return f"Please keep in mind that this is by no means a proper diagnosis, \
treat the result you get from this bot as mere probabilities \
and nothing more.\
\nNow begins your conversation with MedBot!\
The conversational A.I. that will diagnose your illness, provided symptoms. Do start by greeting the bot!\
\n--------------------------------------------------------------------------------------"

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

    bag = [0] * len(words)
    tokenized_inp = nltk.word_tokenize(inp)
    stemmed_inp = generic.stem_words(tokenized_inp)
    bag = [1 if word in stemmed_inp else 0 for word in words]

    return torch.tensor(bag, dtype=torch.float)


class ScrollLabel(qtw.QScrollArea):

    # constructor
    def __init__(self, *args, **kwargs):
        qtw.QScrollArea.__init__(self, *args, **kwargs)

        self.setWidgetResizable(True)

        container_widget = qtw.QWidget(self)
        container_widget.setLayout(qtw.QVBoxLayout())
        
        self.label = qtw.QLabel()
        self.label.setAlignment(qtc.Qt.AlignLeft | qtc.Qt.AlignTop)
        self.label.setWordWrap(True)
        
        container_widget.layout().addWidget(self.label)
        self.setWidget(container_widget)

    def setText(self: ScrollLabel, text: str):
        self.label.setText(text)
        self.repaint()
        qtw.qApp.processEvents()
        
    def text(self: ScrollLabel) -> str:
        return self.label.text()
    
    def setScrollMaximum(self: ScrollLabel) -> str:
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class MainWindow(qtw.QMainWindow):
    def __init__(self: MainWindow) -> None:
        self.handle: str = "You" # User name to show in chat.
        self.user: User = None # User data.
        self.gettingSymptomsFlag: bool = False # `True` if bot is getting symptoms from a user.
        self.endFlag: bool = False # Signifies end of one round of getting symptoms for a user, that is, when probabilities are shown.\
        # If it is set to `False` after being set to `True`, it means input phase for a user has ended, then the GUI resets to welcome a new user.
        self.intents_data: dict[
            list[
                dict[
                    Literal["tag"]: str,
                    Literal["patterns"]: list[str],
                    Literal["responses"]: list[str],
                    Literal["context_set"]: Literal[""]
                ]
            ]
        ] = generic.read_intents_file("json_file/intents.json")
        self.words, self.tags, self.intents_model = load_intents_model(
            pickle_filepath="pickle/intents.pickle", model_filepath="model/intents_model.pt"
        )
        
        super().__init__()

        self.widget = qtw.QWidget()
        
        self.setWindowTitle("MedBot")
        self.setMinimumSize(500, 600)
        
        self.user_input = qtw.QWidget()
        self.user_input.setFixedHeight(30)
        
        self.message_area = ScrollLabel()
        
        self.chat_edit = qtw.QLineEdit()
        self.chat_edit.setFixedHeight(30)
        self.chat_edit.returnPressed.connect(self.UserInputEvent)
        
        self.send_button = qtw.QPushButton("Send")
        self.send_button.setFixedHeight(30)
        self.send_button.clicked.connect(self.UserInputEvent)
        
        self.widget.setLayout(qtw.QFormLayout())
        self.widget.layout().setContentsMargins(0, 5, 0, 0)
        self.widget.layout().setSpacing(5)
        self.user_input.setLayout(qtw.QHBoxLayout())
        self.user_input.layout().setContentsMargins(0, 0, 0, 0)
        self.user_input.layout().setSpacing(0)
        
        self.user_input.layout().addWidget(self.chat_edit)
        self.user_input.layout().addWidget(self.send_button)
        self.widget.layout().addRow(self.message_area)
        self.widget.layout().addRow(self.user_input)
        
        self.setCentralWidget(self.widget)
        
        self.show()
        
        self.chat_edit.setFocus()
        self.reset() # To initialize the message area.
        
    
    def UserInputEvent(self: MainWindow) -> None:
        """
        Method to handle events when the user presses return or "Send" button during input.
        """
        user_message = self.chat_edit.text() # Get user input.
        if user_message == '':
            return
        self.chat_edit.clear() # Clear input area.
        
        self.update_user_message(user_message=user_message)

        if not self.user:
            self.get_user_name(user_message=user_message)

        if self.user:
            if self.gettingSymptomsFlag:
                self.get_symptoms(user_message=user_message)

            else:
                self.user.reset_diagnosis()
                old_messages = self.message_area.text()
                
                if self.endFlag:
                    tag, message_ = self.parse_user_input(user_message=user_message)
                    if tag != "greeting":
                        self.message_area.setText(old_messages+'\n'.join(message_))
                        self.reset()
                        self.endFlag = False
                    else:
                        message_ = [f"\n\nMedBot:"]
                        message_.append(f"Hi, {self.user.first_name}.")
                        message_.append("Can you tell me of the symptoms you're having? ('yes' to stop)")
                        self.message_area.setText(old_messages+'\n'.join(message_))
                        self.gettingSymptomsFlag = True
                else:
                    message_ = ["\n\nMedBot:"]
                    message_.append(f"Hi, {self.user.first_name}.")
                    message_.append("Can you tell me of the symptoms you're having? ('yes' to stop)")
                    self.message_area.setText(old_messages+'\n'.join(message_))
                    self.gettingSymptomsFlag = True
        
        self.message_area.setScrollMaximum()


    def update_user_message(self: MainWindow, user_message: str) -> None:
        """
        Adds the user's message to the message area.
        
        Parameters
        -----------
            user_message (str): User input.
        """
        old_messages = self.message_area.text()
        
        try:
            self.message_area.setText(old_messages+f"\n\n{self.user.first_name}:\n"+user_message)
        except AttributeError:
            self.message_area.setText(old_messages+f"\n\n{self.handle}:\n"+user_message)
            
            
    def get_user_name(self: MainWindow, user_message: str) -> None:
        """
        Chat interface to run until user inputs their name.
        
        Parameters
        -----------
            user_message (str): User input.
        """
        old_messages = self.message_area.text()
        
        if user_message.lower() == "quit":
            return

        if self.handle == "You":
            if (username := generic.read_name(user_message, nlp)) != []:
                self.user = User(username[0][0].split()[0].capitalize())
                return
            
        tag, message_ = self.parse_user_input(user_message=user_message)

        self.message_area.setText(old_messages+'\n'.join(message_))
        


    def parse_user_input(self: MainWindow, user_message: str) -> list[str]:
        """
        Parses user input to discern intent, returns a list of MedBot messages.
        
        Parameters
        -----------
            user_message (str): User input to parse.

        Returns:
            list[str]: List of MedBot messages; each element is to be printed in a different line.
        """
        user_message = bag_of_words_from_input(user_message, words=self.words)
        results = self.intents_model.predict(user_message.view(-1, user_message.shape[0]))
        result_index = torch.argmax(results)
        tag = self.tags[result_index]

        for intent in self.intents_data["intents"]:
            if intent["tag"] == tag:
                responses = intent["responses"]
                break

        message_ = ["\n\nMedBot:"]
        message_.append(f"{random.choice(responses)}")
        
        return tag, message_
        

    def get_symptoms(self: MainWindow, user_message: str) -> None:
        """
        Gets symptoms from user.
        
        Parameters
        -----------
            handle (str): The user handle for the user.
        """
        if user_message.lower() == "yes":
            self.gettingSymptomsFlag = False
        
        old_messages = self.message_area.text()
        
        symptoms, diseases, database_model = load_database_model(
            pickle_filepath="pickle/database.pickle", model_filepath="model/database_model.pt"
        )

        if not self.gettingSymptomsFlag:
        
            user_message = bag_of_words_from_input(self.user.symptoms, words=symptoms)
            database_results = database_model.predict(user_message.view(-1, user_message.shape[0]))

            message_ = ["\n\nMedBot:\nProbabilities:"]
            for disease, result in zip(diseases, *database_results.tolist()):
                message_.append(f"{disease}:  ---  {round(result*100, 2)} %")
            message_.append(f"Highest probability of: {diseases[torch.argmax(database_results)]}")
            
            self.endFlag = True
            self.message_area.setText(old_messages+'\n'.join(message_))
            return

        message_ = ["\n\nMedBot:"]
        
        confirmations = ["Is that all?", "Are you done?", "Move on to the diagnosis?"]
        
        if self.gettingSymptomsFlag:
            symptoms = ' '.join(map(str.strip, user_message.split(',')))
            self.user.symptoms += f' {symptoms}'
        message_.append(f"{random.choice(confirmations)}")
        self.message_area.setText(old_messages+'\n'.join(message_))

    
    def reset(self: MainWindow) -> None:
        self.user = None
        self.message_area.setText(get_welcome_message())


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec())
