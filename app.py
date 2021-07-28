import os
from flask import Flask, render_template
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
from resources.home import Home, Enroll, Verify, Remove

from config import Config

def create_app(config):
    app = Flask(config.APP_NAME)
    app.config.from_object(config)

    from models import db
    db.init_app(app)
    
    api = Api(app)

    api.add_resource(Home,'/')
    api.add_resource(Enroll,'/enroll')
    api.add_resource(Remove,'/remove')
    api.add_resource(Verify,'/verify')

    @app.errorhandler(401)
    def FUN_401(error):
        return render_template("page_401.html"), 401

    @app.errorhandler(403)
    def FUN_403(error):
        return render_template("page_403.html"), 403

    @app.errorhandler(404)
    def FUN_404(error):
        return render_template("page_404.html"), 404

    @app.errorhandler(405)
    def FUN_405(error):
        return render_template("page_405.html"), 405

    @app.errorhandler(413)
    def FUN_413(error):
        return render_template("page_413.html"), 413

    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    return app

app = create_app(Config)

if __name__ == "__main__":
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'])