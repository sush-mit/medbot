import gc
import pickle


import train_on_intents
import train_on_database

def main() -> None:
    """
    Calls functions to parse and train models on intents.json and a disease-symptom database.
    """
    train_on_intents.create_and_train(
        pickle_filepath="pickle/intents.pickle",
        intents_model_filepath="model/intents_model.pt",
        intents_filepath="json_file/intents.json",
        n_epoch=1000,
        batch_size=32,
    )
    
    train_on_database.create_and_train(
        pickle_filepath="pickle/database.pickle",
        database_model_filepath="model/database_model.pt",
        database_filepath="diseases_datasets/disease-symptoms-dataset.sqlite3",
        n_epoch=2000,
        batch_size=32,
    )
    

if __name__ == "__main__":
    main()