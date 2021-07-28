import os
from app import app
from models import db
from config import Config

dbfilepath = Config.SQLALCHEMY_DATABASE_URI[10:]
# Delete database file if it exists currently
if os.path.exists(dbfilepath):
    os.remove(dbfilepath)

db.init_app(app)
# Create the database
with app.app_context():
    db.create_all()
    db.session.commit()