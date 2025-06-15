import uuid

import jwt
from flask import Blueprint, request, jsonify, redirect, send_file
from Ai.requestsHandler import REQUESTS_HANDLER

from Classes.image import Image
from Database.database import database as db
from routes.token_wrapper import token_required
from Classes.user import User

ai_bp = Blueprint('ai_bp', __name__)

@ai_bp.route("/task-cancel/<task_id>", methods=["POST"])
@token_required
def task_cancel(current_user, task_id):
    task_id = int(task_id)
    user = current_user
    if user.current_task_id != task_id:
        return jsonify({"error": "Invalid task_id"}), 400
    result = REQUESTS_HANDLER.cancel(task_id)
    user.current_task_id = -1
    db.session.commit()
    if result is None:
        return jsonify({"error": "Invalid task_id"}), 404
    print("Task canceled")
    return jsonify({"result": f"{result}"}), 200


@ai_bp.route("/task-result/<task_id>", methods=["GET"])
@token_required
def task_result(current_user, task_id):
    task_id = int(task_id)
    user = current_user
    if user.current_task_id != task_id:
        return jsonify({"error": "Invalid task_id"}), 400
    result = REQUESTS_HANDLER.get_task_result(task_id)
    if result is None:
        return jsonify({"error": "Invalid task_id"}), 404
    user.current_task_id = -1
    db.session.commit()
    return send_file(result, mimetype="image/png", as_attachment=True, download_name="result.png")

@ai_bp.route("/task-status/<task_id>", methods=["GET"])
@token_required
def task_status(current_user, task_id):
    task_id = int(task_id)
    user = current_user
    if user.current_task_id != task_id:
        return jsonify({"error": "Invalid task_id"}), 400
    status, result = REQUESTS_HANDLER.get_request_status(task_id)
    if status == 'ERROR':
        return jsonify({"error": "Invalid task_id"}), 404

    if status == 'UNKNOWN_TASK_ID':
        return jsonify({"error": "Unknown task_id"}), 404

    if status == 'CANCELLED':
        return jsonify({"error": "Task canceled"}), 404

    if status == 'PENDING':
        return jsonify({"status": status}), 202

    return jsonify({
        "status": status,
        "message": result[0],
        "progress": result[1],
    })

@ai_bp.route('/outpaint', methods=['POST'])
@token_required
def outpaint(current_user):
    user = current_user
    if user.current_task_id != -1:
        return jsonify({"error": "Task already in progress"}), 400
    image_file = request.files['image']
    task_id = REQUESTS_HANDLER.outpaint(image=image_file)
    user.current_task_id = task_id
    db.session.commit()
    return jsonify({"message": "Outpainting started", "task_id": task_id}), 201

@ai_bp.route('/inpaint', methods=['POST'])
@token_required
def inpaint(current_user):
    user = current_user
    if user.current_task_id != -1:
        return jsonify({"error": "Task already in progress"}), 400
    image_file = request.files['image']
    mask_box = request.form.get('rectBox')
    task_id = REQUESTS_HANDLER.inpaint(image=image_file, mask_box=mask_box)

    user.current_task_id = task_id
    db.session.commit()
    return jsonify({"message": "Inpainting started", "task_id": task_id}), 201

