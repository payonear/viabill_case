import pickle


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)
