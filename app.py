import os
from dotenv import load_dotenv

from Classes.user import User
from Database.database import database
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from routes.auth_routes import auth_bp
from routes.user_routes import user_bp
from routes.ai_routes import ai_bp
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.register_blueprint(auth_bp)
app.register_blueprint(user_bp)
app.register_blueprint(ai_bp)

CORS(app)
database.init_app(app)

with app.app_context():
    #database.drop_all()
    database.create_all()
    User.query.update({User.current_task_id: -1})
    database.session.commit()

if __name__ == '__main__':
    app.run()
