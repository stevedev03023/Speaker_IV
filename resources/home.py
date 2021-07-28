import os
import uuid
import shutil
import numpy as np

from flask import jsonify
from flask import render_template, make_response
from flask_restful import Resource, reqparse

from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from models.voicefootprint import VoicefootprintModel
from speaker_encoder import inference as spk_encoder

from config import Config

ALLOWED_EXTENSIONS = set(['wav'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class Home(Resource):
    def get(self):
        users = []
        users = VoicefootprintModel.user_list()
        return make_response(render_template("home.html", users=users), 200)

class Enroll(Resource):
    if not spk_encoder.is_loaded():
        spk_encoder.load_model(Config.SPEAKER_EMBED_MODEL)

    parser = reqparse.RequestParser()
    parser.add_argument('token', type=str, required=True, help='This field cannot be left blank')
    parser.add_argument('name', type=str, required=True, help='This field cannot be left blank')
    parser.add_argument('audios', type=FileStorage, location='files', action='append', required=True, help='This field cannot be left blank')

    def post(self):
        data = Enroll.parser.parse_args()

        if data['token'] != Config.SECRET_KEY:
            return {'status': 'failed', 'message': 'Invalid token...'}, 200

        if VoicefootprintModel.find_by_name(data['name']):
            return {'status': 'failed', 'message': 'There is already such user, aborting.'}, 200

        erros = {}
        success = True
        wavs = []
        userpath = os.path.join(Config.UPLOAD_FOLDER, data['name'], "master")
        os.makedirs(userpath, exist_ok=True)

        for audiofile in data['audios']:
            if audiofile and allowed_file(audiofile.filename):
                filename = secure_filename(audiofile.filename)
                audiopath = os.path.join(userpath, filename)
                audiofile.save(audiopath)
                wav = spk_encoder.preprocess_wav(audiopath)
                wavs.append(wav)
            else:
                errors[audiofile.filename] = 'Audio file type is not allowed'
                success = False

        if success:
            embed = spk_encoder.embed_speaker(wavs)
            embpath = os.path.join(userpath, Config.SPEAKER_EMBED_FILE)
            np.save(embpath, embed)
            
            user = VoicefootprintModel(data['name'], len(data['audios']))
            user.save_to_db()
            return {'status': 'success', 'message': 'Voice footprint is created successfuly.'}, 200
        else:
            return {'status': 'failed', 'message': jsonify(errors)}, 200
            


class Verify(Resource):
    if not spk_encoder.is_loaded():
        spk_encoder.load_model(Config.SPEAKER_EMBED_MODEL)

    parser = reqparse.RequestParser()
    parser.add_argument('token', type=str, required=True, help='This field cannot be left blank')
    parser.add_argument('name', type=str, required=True, help='This field cannot be left blank')
    parser.add_argument('audio', type=FileStorage, location='files', required=True, help='This field cannot be left blank')

    def post(self):
        data = Verify.parser.parse_args()

        if data['token'] != Config.SECRET_KEY:
            return {'status': 'failed', 'message': 'Invalid token...'}, 200

        user = VoicefootprintModel.find_by_name(data['name'])

        if user == None:
            return {'status': 'failed', 'message': 'There is no such user, aborting.'}, 200

        groundtruthpath = os.path.join(Config.UPLOAD_FOLDER, user.name, "master", Config.SPEAKER_EMBED_FILE)
        groundtruthembed = np.load(groundtruthpath)

        erros = {}
        success = True
        userpath = os.path.join(Config.UPLOAD_FOLDER, data['name'], "verify")
        os.makedirs(userpath, exist_ok=True)
        audiofile = data['audio']
        if audiofile and allowed_file(audiofile.filename):
            filename = str(uuid.uuid4())+"."+audiofile.filename.rsplit('.', 1)[1].lower()
            audiopath = os.path.join(userpath, filename)
            audiofile.save(audiopath)
            wav = spk_encoder.preprocess_wav(audiopath)
            embed = spk_encoder.embed_utterance(wav)
        else:
            errors[audiofile.filename] = 'Audio file type is not allowed'
            success = False

        if success:
            score = (embed*groundtruthembed).sum()
            res = "True" if score > float(Config.SIMILARITY_THRESH) else "False"
            return {'status': 'success', 'message': 'Successfully compaired voice footprint', 'result': res, 'confidence': '%.4f'%score}, 200
        else:
            return {'status': 'failed', 'message': jsonify(errors)}, 200


class Remove(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('token', type=str, required=True, help='This field cannot be left blank')
    parser.add_argument('name', type=str, required=True, help='This field cannot be left blank')

    def post(self):
        data = Remove.parser.parse_args()

        if data['token'] != Config.SECRET_KEY:
            return {'status': 'failed', 'message': 'Invalid token...'}, 200

        user = VoicefootprintModel.find_by_name(data['name'])
        
        if user is not None:
            userpath = os.path.join(Config.UPLOAD_FOLDER, user.name)
            # delete person from database
            user.delete_from_db()
            # if os.path.exists(userpath):
            #     shutil.rmtree(userpath)
            return {'status': 'success', 'message': 'Voice footprint has been removed successfully.'}, 200
        else:
            return {'status': 'failed', 'message': 'There is no such user.'}, 200
