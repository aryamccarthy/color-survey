#
import pandas as pd
import pickle

#
from collections import defaultdict


COORDINATES_FILE = "data/cnum-vhcm-lab-new.txt"
FOCI_FILE = "data/foci-exp.txt"
CACHE_PATH = "data/cached_data.pl"


def make_color_lookup():
    """Get table for converting WCS coordinates to L*a*b*"""
    df = pd.read_table(COORDINATES_FILE)
    df['wcs'] = df["V"] + df["H"].astype(str)
    df.set_index("wcs", inplace=True)
    df = df[["L*", "a*", "b*"]]
    return df


def make_foci_table():
    """Get table of each speaker's focal point responses."""
    column_names = ["Language", "Speaker", "Response", "Abbrev", "Chip"]
    df = pd.read_table(FOCI_FILE, header=None, names=column_names)
    del df["Abbrev"]
    return df

def handle_multiple_foci(lookup, foci):
    """Take the mean as representative of the foci."""
    with_coords = foci.join(lookup, on="Chip", how='inner')
    del with_coords["Chip"]
    return with_coords.groupby(["Language", "Speaker", "Response"]).mean()

def produce_color_lists():
    lookup_table = make_color_lookup()
    foci = make_foci_table()
    data = handle_multiple_foci(lookup_table, foci).reset_index()

    results = defaultdict(dict)
    for l, s in data.groupby(["Language", "Speaker"]).groups:
        subset = data[(data.Language == l) & (data.Speaker == s)]
        results[l][s] = subset[["L*", "a*", "b*"]].values.tolist()
    results = dict(results)
    results = [list(speakers.values()) for speakers in results.values()]
    return results


def get_color_data():
    try:
        with open(CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        data = produce_color_lists()
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(data, f)
        return data


def main():
    for (language, speakers) in get_color_data().items():
        for (speaker, colors) in speakers.items():
            print(colors)


if __name__ == '__main__':
    main()