
def plot_distributions(input_dict:dict, xlabel:str|None=None,
                       stat='count', use_kde:bool=True):
    ncols = 3
    nrows = len(input_dict)
    