import sys
from nimble.deconv import main

if __name__ == "__main__":
    args = sys.argv
    print(args)
    if len(args) > 1:
        main(args)
    else:
        args = [args[0], "auridesi", "06", "120", "1", "100", "4"]
        main(args)
