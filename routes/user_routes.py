import os
import uuid
import io

import jwt
from flask import Blueprint, request, jsonify, redirect

from PIL import Image as PIL_Image
from Classes.image import Image
from Classes.user import User
from Database.database import database as db
from routes.token_wrapper import token_required

user_bp = Blueprint('user_bp', __name__)

@user_bp.route('/user', methods=['GET'])
@token_required
def get_user(current_user):
    return jsonify({"user": current_user.serialize()}), 200

@user_bp.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify({"user": users.serialize()}), 200

@user_bp.route('/user/photo', methods=['GET'])
@token_required
def get_user_photo(current_user):
    data = request.get_json()
    image = Image.query.filter_by(id=data['id'], user_id=current_user.id).first()
    if image is None:
        return jsonify({"message": "Image not found"}), 404
    return jsonify({"photo": image.serialize()}), 200

@user_bp.route('/user/photos', methods=['GET'])
@token_required
def get_user_photos(current_user):
    data = None
    if request.is_json:
        data = request.get_json()

    images = Image.query.filter_by(user_id=current_user.id).order_by(Image.created_at.desc()).all()
    if images is None:
        return jsonify({"message": "Images not found"}), 404
    start, end = 0, len(images)
    if data and 'query_range' in data:
        query_range = data['query_range']
        start = query_range[0]
        end = query_range[1] + 1

    return jsonify({"photos": [photo.serialize() for photo in images[start:end]]}), 200

@user_bp.route('/user/upload/photo', methods=['POST'])
@token_required
def upload_user_photo(current_user):
    """Expects request form"""
    image_file = request.files['image']
    title = request.form.get('title')
    image_file = PIL_Image.open(io.BytesIO(image_file.read()))

    png_path = os.path.join(os.getenv('SAVE_PATH'), f'{len(os.listdir(os.getenv('SAVE_PATH')))}') + '.png'
    new_image = Image(title=title, path=png_path, user_id=current_user.id)
    image_file.save(png_path)

    db.session.add(new_image)
    db.session.commit()

    return {
        'message': 'Image uploaded successfully',
        'image': new_image.serialize()
    }, 201


