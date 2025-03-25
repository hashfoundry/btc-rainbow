from plot import create_plot
from data import get_data


def main(save: bool = True, file_path: str = "bitcoin_rainbow_chart.html"):
    """
    Main function to create and save the Bitcoin rainbow chart.
    
    Args:
        save (bool): Whether to save the chart to a file. Default is True.
        file_path (str): Path to save the HTML file. Default is "bitcoin_rainbow_chart.html".
    """
    # Load data
    raw_data, popt = get_data("data/bitcoin_data.csv")

    # Create plot
    fig = create_plot(raw_data, popt)

    # Save plot as HTML
    if save:
        fig.write_html(
            file_path,
            full_html=True,
            include_plotlyjs='cdn',
            config={
                'displayModeBar': False,
                'responsive': True
            }
        )
        print(f"Chart saved to {file_path}")
    else:
        # Show plot in browser
        fig.show(config={'responsive': True})


if __name__ == "__main__":
    main()
