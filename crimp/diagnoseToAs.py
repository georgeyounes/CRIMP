###############################################################################
# A simple script to create plots of ToA properties, i.e., ToA interval length,
# ToA exposure (livetime), number of counts, count rate, H-test power
# (significance of pulse detection), recduced chi2 (how well the template fit
# the data). These are plotted as a function of ToA number and ToA MJD.
# Plots are created with plotly in simple interactive mode for ease of diagnostics.
#
# Input:
# 1- ToAs : a .txt table of ToA properties (as created with measToAs.py)
# 2- outputFile : name of output interactive html file (default = "ToADiagnosticsPlot".html)
#
# output:
# ToAs : dictionary of ToAs properties
################################################################################

import sys
import argparse
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

sys.dont_write_bytecode = True


################
# Start Script #
################

##############################################################
# Script to create a simple interactive plot of ToA properties
def diagnoseToAs(ToAs, outputFile='ToADiagnosticsPlot'):
    # ToA table as created with measToAs.py
    ToAsProp = pd.read_csv(ToAs, sep='\s+', comment='#')

    tToA_MJD = ToAsProp['ToA_mid'].to_numpy()
    ToAnumber = ToAsProp['ToA'].to_numpy()
    Hpower = ToAsProp['Hpower'].to_numpy()
    redChi2 = ToAsProp['redChi2'].to_numpy()
    ToA_lenInt = ToAsProp['ToA_lenInt'].to_numpy()
    ToA_exp = ToAsProp['ToA_exp'].to_numpy()
    nbr_events = ToAsProp['nbr_events'].to_numpy()
    count_rate = ToAsProp['count_rate'].to_numpy()

    # Creating the figure with plotly - this will redirect to an HTML page - also saved as an HTML file
    fig = make_subplots(rows=6, cols=2,
                        shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.02, vertical_spacing=0.02)
    # vs ToA number
    # ToA interval length and ToA exposure (livetime)
    fig.add_trace(go.Scatter(x=ToAnumber, y=ToA_lenInt / 86400, mode="markers"), row=1, col=1)
    fig.update_yaxes(title_text="ToA interval length (days)", row=1, col=1)
    #
    fig.add_trace(go.Scatter(x=ToAnumber, y=ToA_exp, mode="markers"), row=2, col=1)
    fig.update_yaxes(title_text="ToA exposure (seconds)", row=2, col=1)
    # Number of counts and count rate
    fig.add_trace(go.Scatter(x=ToAnumber, y=nbr_events, mode="markers"), row=3, col=1)
    fig.update_yaxes(title_text="Number of counts", row=3, col=1)
    #
    fig.add_trace(go.Scatter(x=ToAnumber, y=count_rate, mode="markers"), row=4, col=1)
    fig.update_yaxes(title_text="Count rate (/s)", row=4, col=1)
    # H-test and reduced chi2
    fig.add_trace(go.Scatter(x=ToAnumber, y=Hpower, mode="markers"), row=5, col=1)
    fig.update_yaxes(title_text="H-test power", row=5, col=1)
    #
    fig.add_trace(go.Scatter(x=ToAnumber, y=redChi2, mode="markers"), row=6, col=1)
    fig.update_xaxes(title_text="ToA number", row=6, col=1)
    fig.update_yaxes(title_text="Reduced Chi2", row=6, col=1)
    # vs MJD
    # ToA interval length and ToA exposure (livetime)
    fig.add_trace(go.Scatter(x=tToA_MJD, y=ToA_lenInt / 86400, mode="markers"), row=1, col=2)
    #
    fig.add_trace(go.Scatter(x=tToA_MJD, y=ToA_exp, mode="markers"), row=2, col=2)
    # Number of counts and count rate
    fig.add_trace(go.Scatter(x=tToA_MJD, y=nbr_events, mode="markers"), row=3, col=2)
    #
    fig.add_trace(go.Scatter(x=tToA_MJD, y=count_rate, mode="markers"), row=4, col=2)
    # H-test and reduced chi2
    fig.add_trace(go.Scatter(x=tToA_MJD, y=Hpower, mode="markers"), row=5, col=2)
    #
    fig.add_trace(go.Scatter(x=tToA_MJD, y=redChi2, mode="markers"), row=6, col=2)
    fig.update_xaxes(title_text="Days (MJD)", row=6, col=2)
    #
    fig.update_layout(height=1600, width=1600, title_text="ToA properties for file " + ToAs, showlegend=False,
                      font=dict(size=14))
    #
    fig.show()
    fig.write_html('./' + outputFile + '.html')

    return ToAsProp


def main():
    parser = argparse.ArgumentParser(description="Script to measure ToAs from event file")
    parser.add_argument("ToAs", help=".txt file of phase shifts created with measToAs.py, e.g., ToAs.txt", type=str)
    parser.add_argument("-of", "--outputFile",
                        help="name of output ToA diagnostics file (default = ToADiagnosticsPlot(.html))", type=str,
                        default='ToADiagnosticsPlot')
    args = parser.parse_args()

    diagnoseToAs(args.ToAs, args.outputFile)


if __name__ == '__main__':
    main()
