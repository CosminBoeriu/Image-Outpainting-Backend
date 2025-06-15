import os
import uuid
from functools import wraps

import jwt
from flask import Blueprint, request, jsonify, redirect
from Database.database import database as db

from Classes.user import User

auth_bp = Blueprint('auth_bp', __name__)


def validate_email_and_password(email, password):
    return True

@auth_bp.route('/register', methods=['POST'])
def register():
    print(User.query.all())
    if request.method == 'POST':
        data = request.get_json()
        email = data['email']
        username = data['username']
        password = data['password']

        is_validated = validate_email_and_password(email, password)
        if is_validated is not True:
            return jsonify({'message': 'Invalid data', }), 400

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({"message": "Email already exists."}), 400

        new_user = User(email=email, username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        token = jwt.encode({'id': new_user.id}, os.getenv('SECRET_KEY'), algorithm = "HS256")

        return jsonify({"user": new_user.serialize(), "token": token, "username": username}), 200

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    #TODO: email and password verifications
    try:
        data = request.get_json()
        if not data:
            return jsonify({'message': 'Missing data'}), 400
        email, password = data.get('email'), data.get('password')
        is_validated = validate_email_and_password(email, password)
        if is_validated is not True:
            return jsonify({"message":'Invalid data'}), 400
        user = User.query.filter_by(email=data.get('email')).first()
        if user:
            try:
                if not user.check_password(password):
                    return jsonify({"message": "Invalid email or password"}), 401
                token = jwt.encode({"id": user.id},os.getenv('SECRET_KEY'),algorithm="HS256")
                return jsonify({"user": user.serialize(), "token": token, "username": user.username}), 200
            except Exception as e:
                return jsonify({'message': str(e)}), 500
        return jsonify({"message": "Unknown email address!"}), 401
    except Exception as e:
        return jsonify({'message': str(e)}), 500

