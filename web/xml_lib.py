import os
from xml.dom.minidom import parse
import xml.dom.minidom

base = os.path.abspath(".")
xml_file = os.path.join(base, 'web', 'file', 'movies.xml')

def read_movies():
    DOMTree = xml.dom.minidom.parse(xml_file)
    root = DOMTree.documentElement
    movies = root.getElementsByTagName('movie')

    movie_arr = []
    for movie in movies:
        movie_dic = {}
        movie_dic['file'] = movie.getAttribute('file')
        movie_dic['title'] = movie.getAttribute('title')
        movie_arr.append(movie_dic)

    return movie_arr