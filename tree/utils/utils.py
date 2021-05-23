def is_unique(array):
    unique = array[0]
    for item in array:
        if item != unique:
            return False
    return True

def unique_count(array):
    """
    返回序列 `array` 中的标签及其对应数量
    Parameters
    ----------
    array : array-like, dtype=double
        array
    """
    result = {}
    for item in array:
        result[item] = result.get(item, 0) + 1
    return result
