import os
import uuid
from functools import wraps

import jwt
from flask import Blueprint, request, jsonify, redirect
from Database.database import database as db
from Classes.user import User

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', None)
        if not auth_header:
            return jsonify({"message": "Token is missing"}), 401

        parts = auth_header.split()

        # Expecting header of the form "Bearer <token>"
        if parts[0].lower() != 'bearer' or len(parts) != 2:
            return jsonify({"message": "Invalid Authorization header"}), 401

        token = parts[1]
        try:
            data=jwt.decode(token, os.getenv('SECRET_KEY'), algorithms=["HS256"])
            current_user=User.query.filter_by(id=data['id']).first()
            if current_user is None:
                return {
                "message": "Invalid Authentication token!",
            }, 401
        except Exception as e:
            return {
                "message": str(e),
            }, 500

        return f(current_user, *args, **kwargs)

    return decorated