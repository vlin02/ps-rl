import enum
import time

class TermColors(enum.Enum):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    ORANGE = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def infx(*args, color=TermColors.RED):
    print(color.value, *args, TermColors.ENDC.value)

def get_module_name(module):
    return module.__name__.split('.')[-1]

def get_date_file_name():
    return time.strftime("%m-%d %H:%M:%S")