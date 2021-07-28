import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class Config(object):
    APP_NAME = os.environ.get("APP_NAME")
    FLASK_ENV = os.environ.get("FLASK_ENV")

    PORT = os.environ.get("PORT")
    HOST = os.environ.get("HOST")
    DEBUG = bool(os.environ.get("DEGUB"))

    SQLALCHEMY_DATABASE_URI = os.environ.get("SQLALCHEMY_DATABASE_URI")
    SQLALCHEMY_TRACK_MODIFICATIONS = os.environ.get("SQLALCHEMY_TRACK_MODIFICATIONS")
    
    SECRET_KEY = os.environ.get("SECRET_KEY")

    SPEAKER_EMBED_MODEL = os.environ.get("SPEAKER_EMBED_MODEL")
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER")
    SPEAKER_EMBED_FILE = os.environ.get("SPEAKER_EMBED_FILE")
    SIMILARITY_THRESH = float(os.environ.get("SIMILARITY_THRESH"))
