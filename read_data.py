#
import pandas as pd
import pickle
from typing import List

#
from collections import Counter, defaultdict

#
import matplotlib.pyplot as plt
import seaborn as sns


COORDINATES_FILE = "data/cnum-vhcm-lab-new.txt"
FOCI_FILE = "data/foci-exp.txt"
CACHE_PATH = "data/cached_data.pl"


def make_color_lookup() -> pd.DataFrame:
    """Get table for converting WCS coordinates to L*a*b*"""
    df = pd.read_table(COORDINATES_FILE)
    df['wcs'] = df["V"] + df["H"].astype(str)
    df.set_index("wcs", inplace=True)
    df = df[["L*", "a*", "b*"]]
    return df


def make_foci_table() -> pd.DataFrame:
    """Get table of each speaker's focal point responses."""
    column_names = ["Language", "Speaker", "Response", "Abbrev", "Chip"]
    df = pd.read_table(FOCI_FILE, header=None, names=column_names)
    del df["Abbrev"]
    return df

def handle_multiple_foci(lookup, foci) -> pd.DataFrame:
    """Take the mean as representative of the foci."""
    with_coords = foci.join(lookup, on="Chip", how='inner')
    del with_coords["Chip"]
    return with_coords.groupby(["Language", "Speaker", "Response"]).mean()

def produce_color_lists() -> List[List[List[float]]]:
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


def plot_inventory_sizes():
    lengths = []
    for language in get_color_data():
        for inventory in language:
            print(len(inventory))
            lengths.append(len(inventory))
            # break
    lengths_c = Counter(lengths)
    cats, counts = zip(*sorted(lengths_c.items()))
    cats, counts = list(cats), list(counts)
    if 21 in cats:
        cats.insert(-1, 20)
        counts.insert(-1, 0)
    sns.set_context('paper')
    sns.barplot(cats, counts, color='royalblue')
    sns.despine(left=True, bottom=True)
    ax = plt.gca()
    ax.yaxis.grid(True, color='w')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.savefig(f"figures/sizes.pdf", bbox_inches='tight')
    plt.show()


def main():
    plot_inventory_sizes()

if __name__ == '__main__':
    main()