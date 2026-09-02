"""Small shared helpers for the figures this project draws.

Every plot gets a generation time stamped into its bottom-right corner, so a
screenshot taken out of context still says which run it came from.
"""

from datetime import datetime


def timestamp_figure(fig, when = None, label = 'Generated', color = '0.45'):
    """Stamp a time into the bottom-right corner of `fig`.

    `when` defaults to now; pass the run's start time where that is the more
    useful moment. `color` is lightened by callers whose figures are dark.
    Returns the Text artist so a live display can update it in place.
    """
    when = when or datetime.now()
    # Managed layouts fill the figure to its edge, so reserve a strip for the stamp
    # instead of letting it land on top of the bottom row's x label.
    engine = getattr(fig, 'get_layout_engine', lambda: None)()
    if engine is not None:
        try:
            engine.set(rect = (0, 0.028, 1, 0.972))  # left, bottom, width, height
        except (AttributeError, TypeError):
            pass
    return fig.text(0.995, 0.005, f'{label}: {when.strftime("%Y-%m-%d %H:%M:%S")}',
                    ha = 'right', va = 'bottom', fontsize = 7, color = color)
