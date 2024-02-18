import glob

def get_word2ix(path = "./../vocab.txt"):
    """ Generates a mapping given a vocabulary file. 
    Input
    -------------
    path: str or pathlib.Path. Relative path to the vocabulary file. 

    Output
    -------------
    word2ix: dict. Dictionary mapping words to unique IDs. Keys are words and 
                values are the indices.
    """
    word2ix = {}
    with open(path) as f:
        data = f.readlines()
        for line in data:
            word2ix[line.split("\t")[1].strip()] = int(line.split("\t")[0])
    
    return word2ix


def get_files(path):
    """ Returns a list of text files in the 'path' directory.
    Input
    ------------
    path: str or pathlib.Path. Directory path to load files from. 

    Output
    -----------
    file_list: List. List of paths to text files
    """
    file_list =  list(glob.glob(f"{path}/*.txt"))
    return file_list


def process_data(files, context_window, word2ix):
    """ Returns the processed data. Processing involves reading data from
    the files, converting the words to appropriate indices, mapping OOV words
    to the [UNK] token and padding appropriately.
    Inputs
    -----------
    files: List. List of files to be processed. Can be the list
            returned by the `get_files()` method.
    context_window: int. Size of the context window. Size is the amount
            of words considered as context either to the left or right of a word
    word2ix: dict. Mapping from word to a unique index. Can be the dict returned by
                the `get_word2ix` method

    Output
    ----------
    data: List[List[int]]. Each list corresponds to a file and the set of indices
            for the contents of the file.
    """
    data = []
    for file in files:
        file_data = [word2ix["[PAD]"]]*context_window
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                if line.strip() not in word2ix.keys():
                    file_data.append(word2ix["[UNK]"])
                else:
                    file_data.append(word2ix[line.strip()])          
            file_data.extend([word2ix["[PAD]"]]*context_window)
            data.append(file_data.copy())
    return data