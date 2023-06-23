import io
import torch
import pickle
import json


class CPU_Unpickler(pickle.Unpickler):
    """ Helper function to make stuff stored on gpu loadable on cpu. """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def pickle_dump(x, path):
    """ Dumps given data to file path. """
    pickle.dump(x, open(path, "wb"))


def pickle_load(path):
    """ Loads pickled data from given file path. """
    return CPU_Unpickler(open(path, "rb")).load()


def npyfy(x):
    """ Detaches tensor, puts it on to cpu and returns numpy array. """
    return x.detach().cpu().numpy()


def json_dump(x, path):
    """ Dumps given data to json file. """
    with open(path, "w") as outfile:
        json.dump(x, outfile)


def json_load(path):
    """ Reads data from json file. """
    with open(path, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
    return json_object
