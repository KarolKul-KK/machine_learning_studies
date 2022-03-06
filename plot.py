import pandas as pd
import seaborn as sns
import altair as alt
import numpy as np

import os


def make_plot_from_df(df: pd.DataFrame) -> alt.vegalite.v4.api.LayerChart:

    degree_list = [1, 3, 5]

    base = alt.Chart(df).mark_circle(size=60).encode(
        x = 'flipper_length_mm',
        y = 'body_mass_g',
        color='species',
        tooltip=['species', 'flipper_length_mm', 'body_mass_g']
    )

    polynomial_fit = [
        base.transform_regression(
            "flipper_length_mm", "body_mass_g", method="poly", order=order, as_=["flipper_length_mm", str(order)]
        )
        .mark_line()
        .transform_fold([str(order)], as_=["degree", "body_mass_g"])
        .encode(alt.Color("degree:N"))
        for order in degree_list
    ]

    plot = alt.layer(base, *polynomial_fit).interactive()

    alt.layer(base, *polynomial_fit).save('plot.html')

    return plot


def main():

    penguins = sns.load_dataset('penguins')
    make_plot_from_df(penguins)


if __name__ == "__main__":

    main()