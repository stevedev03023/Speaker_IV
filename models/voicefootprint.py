from flask_sqlalchemy import SQLAlchemy
from models import db
from config import Config

class VoicefootprintModel(db.Model):
    
    __tablename__ = 'voicefootprint'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    num_audios = db.Column(db.Integer, nullable=False)

    def __init__(self, _name, _num_audios):
        self.name = _name
        self.num_audios = _num_audios

    def save_to_db(self):
        db.session.add(self)
        db.session.commit()

    def delete_from_db(self):
        db.session.delete(self)
        db.session.commit()
        
    @classmethod
    def find_by_name(cls, _name):
        return cls.query.filter_by(name=_name).first()

    @classmethod
    def find_by_id(cls, _id):
        return cls.query.filter_by(id=_id).first()

    @classmethod
    def user_list(cls):
        return [(user.id, user.name, user.num_audios) for user in cls.query.all()]
