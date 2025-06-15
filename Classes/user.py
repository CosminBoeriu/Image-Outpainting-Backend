from Database.database import database as db
from werkzeug.security import generate_password_hash, check_password_hash
from Classes.image import Image


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # → user.id INTEGER PRIMARY KEY
    email = db.Column(db.String(120), unique=True)
    username = db.Column(db.String(120))
    password_hash = db.Column(db.String(128), nullable=False)
    current_task_id = db.Column(db.Integer, default=-1, nullable=False)

    def __init__(self, email, password, **kwargs):
        super().__init__(**kwargs)
        self.email = email
        self.password_hash = generate_password_hash(password)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def serialize(self):
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "current_task_id": self.current_task_id
        }

