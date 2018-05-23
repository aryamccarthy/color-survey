# pylint: disable=missing-docstring, invalid-name

#
import pandas as pd


def make_color_lookup(file):
    """Get table for converting WCS coordinates to L*a*b*"""
    df = pd.read_table(file)
    df["wcs"] = df["V"] + df["H"].astype(str)  # I36 = I + 36
    df.set_index("wcs", inplace=True)
    df = df[["L*", "a*", "b*"]]
    return df


def make_foci_table(file):
    column_names = ["Language", "Speaker", "Response", "Abbrev", "Chip"]
    df = pd.read_table(file, header=None, names=column_names)
    del df["Abbrev"]
    return df


def handle_multiple_foci(lookup, foci):
    with_coords = foci.join(lookup, on="Chip", how='inner')
    del with_coords["Chip"]
    return with_coords.groupby(["Language", "Speaker", "Response"]).mean()


def produce_color_lists(mapping_file="data/cnum-vhcm-lab-new.txt",
                        foci_file="data/foci-exp.txt"):
    lookup_table = make_color_lookup(mapping_file)
    foci = make_foci_table(foci_file)
    data = handle_multiple_foci(lookup_table, foci).reset_index()

    one_speaker_per_language = data[data["Speaker"] == 1]
    languages = one_speaker_per_language.groupby(["Language"]).groups
    for language in languages:
        subset = one_speaker_per_language[one_speaker_per_language.Language == language]
        yield subset[["L*", "a*", "b*"]].values.tolist()





def main():
    for color_list in produce_color_lists():
        print(len(color_list))
    # print(one_speaker_per_language[["L*", "a*", "b*"]].values.tolist())

if __name__ == '__main__':
    main()
