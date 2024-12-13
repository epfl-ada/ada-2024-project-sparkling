import os
PLOTS_PATH = "plots"

def save_plot(fig, figure_name):
    """
    Given a plotly figure and a figure name, save it in the right file

    Argument:
        - fig: Plotly figure
        - figure_name: Figure name (saved as: <figure_name>.html)
    """
    os.makedirs(PLOTS_PATH, exist_ok=True)

    filepath = os.path.join(PLOTS_PATH, f"{figure_name}.html")
    fig.write_html(filepath)
    print(f"Figure saved in {filepath}")