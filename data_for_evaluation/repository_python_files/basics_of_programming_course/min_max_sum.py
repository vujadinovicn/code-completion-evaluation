def minimum(array):
    minimal = array[0]
    for element in array:
        if element<minimal:
            minimal = element
    return minimal

def maximum(array):
    maximal = array[0]
    for element in array:
        if element > maximal:
            maximal = element
    return maximal

def sum(array):
    sum = 0
    for element in array:
        sum += element
    return sum