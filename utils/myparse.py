from custom_types import *
import sys


# minimal argparse
def parse(parse_dict: Dict[str, Dict[str, Any]]):
    args = sys.argv[1:]
    args_dict = {args[i]: args[i + 1] for i in range(0, len(args), 2)}
    out_dict = {}
    for item in parse_dict:
        hint = parse_dict[item]['type']
        if item in args_dict:
            out_dict[item[2:]] = hint(args_dict[item])
        else:
            out_dict[item[2:]] = parse_dict[item]['default']
        if 'options' in parse_dict[item] and out_dict[item[2:]] not in parse_dict[item]['options']:
            raise ValueError(f'Expected {item} to be in {str(parse_dict[item]["options"])}')
    return out_dict
