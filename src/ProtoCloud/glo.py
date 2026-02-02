"""
Help setup global variables
"""
_global_dict: dict[str, object] = {} 
 
def set_value(key, value):
    global _global_dict
    _global_dict[key] = value
 
 
def get_value(key):
    try:
        return _global_dict[key]
    except KeyError as err:
        raise KeyError(f"Read {key} failed") from err