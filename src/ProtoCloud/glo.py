"""
Help setup global variables
"""

_global_dict: dict[str, object] = {} 

def reset():
    """Optional: clear all stored globals."""
    _global_dict.clear()
 
def set_value(key, value):
    global _global_dict
    _global_dict[key] = value
 
 
def get_value(key):
    try:
        return _global_dict[key]
    except KeyError:
        raise KeyError('Read ' + key + ' Failed\r\n')