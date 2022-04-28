import os
from flask import Flask, redirect, send_from_directory, url_for, request, render_template, abort
from werkzeug.utils import secure_filename
from Code.main import auto_subtitle
import argparse

base = os.path.abspath(os.path.join(os.path.abspath("."), ".."))
template_path = os.path.join(base, "web", "templates")
web = Flask(__name__, template_folder=template_path)
web.config['UPLOAD_EXTENSIONS'] = ['.mp4']
web.config['UPLOAD_PATH'] = os.path.join(base, "Data", "videos", "original")
web.config["SUBTITLED_PATH"] = os.path.join(base, "Data", "videos", "subtitled")

@web.route('/')
def hello_world():
    return "Hello World!"

@web.route('/movieList/<filename>')
def movie_list(filename):
    return render_template("movie_list.html", movie = filename)

@web.route('/subtitled_movie/<filename>')
def subtitled_movie(filename):
    return render_template("subtitled_movie.html", movie = filename)

@web.route('/upload')
def upload():
    movie_list = os.listdir(web.config["UPLOAD_PATH"])
    return render_template('upload.html', movies = movie_list)

@web.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        for uploaded_file in request.files.getlist('file'):
            if uploaded_file.filename != "":
                file_ext = os.path.splitext(uploaded_file.filename)[1]
                if  file_ext not in web.config['UPLOAD_EXTENSIONS']:
                    abort(400)
                uploaded_file.save( os.path.join(web.config["UPLOAD_PATH"], \
                                    secure_filename(uploaded_file.filename)))
                return redirect(url_for("upload"))

@web.route('/upload/<filename>')
def uploaded(filename):
    return send_from_directory(web.config["UPLOAD_PATH"], filename)

@web.route('/subtitled/<filename>')
def subtitled(filename):
    return send_from_directory(web.config["SUBTITLED_PATH"], filename)

@web.route('/movie/<filename>', methods = ['GET', 'POST'])
def add_subtitle(filename):
    if request.method == "POST":
        auto_subtitle(filename, base)
    file, ext = filename.split(".")
    return redirect(url_for("subtitled_movie", filename = f"{file}_subtitled.{ext}"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=str)
    parser.add_argument('--address',
                        type=str)
    arg = parser.parse_args()
    web.run(host = arg.address, port = arg.port, debug = True)