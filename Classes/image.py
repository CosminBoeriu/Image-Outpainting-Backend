import os
from datetime import datetime

from Database.database import database as db
from PIL import Image as PIL_Image

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String(200))
    title = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.now())
    updated_at = db.Column(db.DateTime, default=datetime.now(), onupdate=datetime.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def get_image(self):
        return PIL_Image.open(self.path)

    def serialize(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "path": os.path.abspath(self.path)
        }

