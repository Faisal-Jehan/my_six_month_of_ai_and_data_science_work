from flask import Flask 

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World! WOW what a fantastic app this was</p>"