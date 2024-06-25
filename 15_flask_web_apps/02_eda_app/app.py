import matplotlib.pyplot as plt

def create_plots():
    # Example plot
    plt.figure(figsize=(12, 5))
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.title('Example Plot')
    plt.savefig('static/plot1.png')
    plt.close()

    # Another example plot
    plt.figure(figsize=(6, 5))
    plt.plot([1, 2, 3], [6, 5, 4])
    plt.title('Another Plot')
    plt.savefig('static/plot2.png')
    plt.close()

if __name__ == "__main__":
    create_plots()

from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)


from flask import Flask, render_template_string
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    try:
        # Load the iris dataset
        iris = sns.load_dataset('iris')
        
        # Perform basic EDA
        summary_stats = iris.describe().to_html()
        species_counts = iris['species'].value_counts().to_frame().to_html()
        
        # Plotting
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Pairplot
        pairplot_fig = sns.pairplot(iris, hue='species')
        pairplot_img = io.BytesIO()
        pairplot_fig.savefig(pairplot_img, format='png')
        pairplot_img.seek(0)
        pairplot_url = base64.b64encode(pairplot_img.getvalue()).decode()
        plt.close(pairplot_fig.fig)
        
        # Plot 2: Species count plot
        plt.figure(figsize=(6, 5))
        species_count_fig = sns.countplot(data=iris, x='species')
        species_count_img = io.BytesIO()
        plt.savefig(species_count_img, format='png')
        species_count_img.seek(0)
        species_count_url = base64.b64encode(species_count_img.getvalue()).decode()
        plt.close()
        
        # HTML template
        html_template = '''
        <html>
            <head><title>Iris Dataset EDA</title></head>
            <body>
                <h1>Iris Dataset EDA</h1>
                <h2>Summary Statistics</h2>
                {{ summary_stats|safe }}
                <h2>Species Counts</h2>
                {{ species_counts|safe }}
                <h2>Plots</h2>
                <h3>Pairplot</h3>
                <img src="data:image/png;base64,{{ pairplot_url }}">
                <h3>Species Count Plot</h3>
                <img src="data:image/png;base64,{{ species_count_url }}">
            </body>
        </html>
        '''
        
        return render_template_string(html_template, summary_stats=summary_stats, species_counts=species_counts, pairplot_url=pairplot_url, species_count_url=species_count_url)
    except Exception as e:
        logging.error("Error occurred: %s", e)
        return "An error occurred: " + str(e)

if __name__ == '__main__':
    app.run(debug=True)
