import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Define a function that performs interactive data visualization using Plotly Express
def plot_financial_data(df, title):
    fig = px.line(title = title)
    
    for i in df.columns[1:]:
        fig.add_scatter(x = df['Date'], y = df[i], name = i)
        fig.update_traces(line_width = 2)
        fig.update_layout({'plot_bgcolor': "white"})
    fig.show()

def plot_histogram_data(df):
    fig = px.histogram(df.drop(columns = ['Date']))
    fig.update_layout({'plot_bgcolor': "white"})
    fig.show()

def plot_heatmap_data(df):
    plt.figure(figsize = (10, 8))
    sns.heatmap(df.drop(columns = ['Date']).corr(), annot = True)
