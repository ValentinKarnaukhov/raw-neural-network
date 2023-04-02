import pickle


def serialize(file_name, data):
    with open(file_name, "wb") as output_file:
        pickle.dump(data, output_file)


def deserialize(file_name):
    with open(file_name, "rb") as input_file:
        return pickle.load(input_file)
