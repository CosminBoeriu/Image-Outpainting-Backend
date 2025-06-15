import os
import uuid

import jwt
from flask import Blueprint, request, jsonify, redirect

from Classes.image import Image
from Database.database import database as db
from routes.token_wrapper import token_required

image_bp = Blueprint('user_bp', __name__)

@image_bp.route('/photo', methods=['GET'])
#TODO delete this function because it does not require auth
def get_photo():
    data = request.get_json()
    image = Image.query.filter_by(id=data['id']).first()
    return jsonify({"photo": image.serialize()}), 200




