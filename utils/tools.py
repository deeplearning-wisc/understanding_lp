import torch

def get_shape_of_tensor(tensor):
    shape = []
    e = tensor
    # while e is not a tensor
    while not isinstance(e, torch.Tensor):
        shape.append(len(e)) # batch size
        e = e[0]
    shape.extend(list(e.size())) # shape of the tensor
    return shape

def get_heterogenous_stack(list_of_lists):
    max_length = max(len(lst) for lst in list_of_lists)
    return [lst + [None] * (max_length - len(lst)) for lst in list_of_lists]

def get_column_average(heterogenous_stack):
    column_average = []
    for i in range(len(heterogenous_stack[0])):
        sum = 0
        num = 0
        for j in range(len(heterogenous_stack)):
            if heterogenous_stack[j][i] is not None:
                sum += heterogenous_stack[j][i]
                num += 1
        column_average.append(sum / num)
    return column_average

def concat_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]