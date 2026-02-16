from flask import Flask
from flask_bootstrap import Bootstrap
import os

UPLOAD_FOLDER = os.getcwd()

app = Flask(__name__)
Bootstrap(app)
app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 * 1024

