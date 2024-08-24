from conductor.reports.models import Timeline
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def draw_timeline(timeline: Timeline, timeline_path: str) -> None:
    """_summary_

    Args:
        timeline (Timeline): _description_
        timeline_path (str): _description_
    """

    # Prepare data for plotting
    dates = []
    events = []
    for event in timeline.events:
        # Construct the date string, using '01' as a default value for missing month/day
        date_str = f"{event.year}-{event.month or '01'}-{event.day or '01'}"
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        dates.append(date_obj)
        events.append(event.event)

    # Plotting the timeline on a single line
    fig, ax = plt.subplots(figsize=(10, 3))

    # Draw a horizontal line
    ax.hlines(1, min(dates), max(dates), color="black", linewidth=1)

    # Plot the ticks on the line
    ax.plot(dates, [1] * len(dates), "|", markersize=10, color="blue")

    # Annotate the events above the ticks
    # Annotate the events alternating above and below the line
    for i, (date, event) in enumerate(zip(dates, events)):
        y_offset = 1.03 if i % 2 == 0 else 0.97
        ax.text(
            date,
            y_offset,
            event,
            rotation=0,
            ha="right",
            fontsize=10,
            verticalalignment="bottom" if i % 2 == 0 else "top",
        )

    # Formatting the plot
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.yaxis.set_visible(False)  # Hide y-axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # set up tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Timeline of Events")
    plt.tight_layout()
    plt.savefig(timeline_path, dpi=300, bbox_inches="tight")
    return True
