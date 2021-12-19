"""
Download xlsx file from the given url and save it as a csv file
at the given location

Usage: src/download_data.py --url=<url> --out_file=<out_file>

Options:
--url=<url>                 Url of the excel (xlsx) file
--out_file=<out_file>       Output location of the csv file
"""

from docopt import docopt
import os
import pandas as pd

opt = docopt(__doc__)

def main(url, out_file):
    
    data = pd.read_excel(url, header=1)

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    
    data.to_csv(out_file, index=False, encoding="utf-8")

if __name__ == "__main__":
    url = opt["--url"]
    out_file = opt["--out_file"]

    main(url, out_file)