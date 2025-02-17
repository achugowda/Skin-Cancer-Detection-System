"""
    This code saves a trained model to a file using pickle and later loads it for reuse.
"""
import pickle

filename = "trained_model.sav"
pickle.dump(model, open(filename, 'wb'))

#loading the saved model
loaded_model = pickle.load(open(filename, 'rb'))