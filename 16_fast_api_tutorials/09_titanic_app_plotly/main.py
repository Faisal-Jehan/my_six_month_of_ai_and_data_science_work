from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import seaborn as sns
import pandas as pd
import plotly.express as px
from plotly.io import to_html

app = FastAPI() # Create a FastAPI instance

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Perform a simple data transformation
survival_rate = df.groupby("class")["survived"].mean().reset_index()

@app.get("/")
async def root():
    return {"message": "Welcome to the Titanic API, we are learning on Codanics.com"}

@app.get("/survival_rate")
async def get_survival_rate():
    return survival_rate.to_dict(orient="records")

@app.get("/survival_rate_plotly", response_class=HTMLResponse)
async def survival_rate_plotly():
    # Generate the plot with Plotly
    fig = px.bar(survival_rate, x='class', y='survived', 
               # color='sex',
                title="Survival Rate by Class with Plotly")

    # Convert the plot to HTML
    plot_div = to_html(fig, full_html=False)

    # Create the full HTML document
    html_content = f"""
    <html>
        <head>
            <title>Survival Rate by Class based on Gender (Plotly)</title>
        <head>
        <body>
            {plot_div}
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)
