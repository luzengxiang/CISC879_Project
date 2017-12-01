
from lib.crop_image import import_eps
import argparse
import numpy as np
import os
import sys
if __name__ == '__main__':


    parser = argparse.ArgumentParser(description = "data_process based on Zone")
    parser.add_argument('Zone', type=int,help='Positional zone number:int')
    args = parser.parse_args()


    if os.path.exists("../data/processed/Zone_%d" % args.Zone) == False:
        os.mkdir("../data/processed/Zone_%d" % args.Zone)


    X, Y = import_eps("../data", '../data/stage1_labels.csv', args.Zone)

    sys.stderr.write("Saving Data:" + '\n')
    np.save("../data/processed/Zone_%d/X" % args.Zone,X)
    np.save("../data/processed/Zone_%d/Y" % args.Zone,Y)