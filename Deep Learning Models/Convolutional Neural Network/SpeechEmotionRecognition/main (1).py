import os

# import magic
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import cnn

ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif", "wav", "mp3", "mp4"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def upload_form():
    return render_template("upload2.html")


@app.route("/", methods=["POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected for uploading")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            got_emotion = cnn.web_test_label(filepath)

            flash(f"Name of File Uploaded: {str(file.filename)}")
            flash(f"Emotion Detected: {got_emotion}")
            return redirect("/")
        else:
            flash("Allowed file types are txt, pdf, png, jpg, jpeg, gif, mp3, mp4, wav")
            return redirect(request.url)


if __name__ == "__main__":
    app.run()
