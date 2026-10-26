import os
import io
import base64
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from inference import predict, MODEL_NAME, THRESHOLD
from database import init_db, log_inference, get_all_inferences

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "visionguard-secret-key-23L-0540")

# Task 4.2: Maximum file upload size limit (5 MB)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "5"))
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize database
init_db()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    preview_img_data = None
    error_message = None

    if request.method == "POST":
        # 1. Validation: File presence check
        if "file" not in request.files:
            error_message = "No file part in the upload request."
            return render_template("index.html", error=error_message)

        file = request.files["file"]
        if file.filename == "":
            error_message = "Please select an image file before clicking Analyze."
            return render_template("index.html", error=error_message)

        # 2. Validation: File extension check (reject non-images)
        if not allowed_file(file.filename):
            error_message = f"Unsupported file type '{file.filename}'. Allowed formats: PNG, JPG, JPEG, WEBP, BMP."
            return render_template("index.html", error=error_message)

        # Sanitize filename (Sub-task 4.2)
        safe_name = secure_filename(file.filename)

        # 3. Read in-memory: DO NOT save file to disk (Sub-task 4.2)
        file_bytes = file.read()

        # 4. In-memory validation & prediction
        try:
            prediction_result = predict(file_bytes)
            
            # Encode image to base64 for browser display preview without disk storage
            b64_img = base64.b64encode(file_bytes).decode("utf-8")
            preview_img_data = f"data:image/jpeg;base64,{b64_img}"

            # 5. Task 5: Permanent Database Trail
            log_inference(
                filename=safe_name,
                verdict=prediction_result["verdict"],
                confidence=prediction_result["confidence"],
                raw_score=prediction_result["raw_score"],
                model_name=MODEL_NAME,
                threshold=THRESHOLD
            )
        except Exception as e:
            error_message = f"Validation Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction_result,
        preview_img=preview_img_data,
        error=error_message,
        model_name=MODEL_NAME,
        threshold=THRESHOLD
    )

@app.route("/history")
def history():
    """Task 5: Display past search history from SQLite."""
    records = get_all_inferences(limit=50)
    return render_template("history.html", records=records, model_name=MODEL_NAME)

@app.route("/health")
def health():
    """Sub-task 4.2 & 4.4: Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "VisionGuard Web App",
        "model_name": MODEL_NAME,
        "threshold": THRESHOLD,
        "max_upload_size_mb": MAX_FILE_SIZE_MB
    }), 200

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic inference (Task 7 & 8)."""
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in request body."}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format."}), 400

    safe_name = secure_filename(file.filename)
    file_bytes = file.read()

    try:
        res = predict(file_bytes)
        log_inference(
            filename=safe_name,
            verdict=res["verdict"],
            confidence=res["confidence"],
            raw_score=res["raw_score"],
            model_name=MODEL_NAME,
            threshold=THRESHOLD
        )
        return jsonify({
            "status": "success",
            "filename": safe_name,
            **res
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 422

# Error handlers for friendly user display
@app.errorhandler(413)
def request_entity_too_large(error):
    """Sub-task 4.2: Friendly error for oversized uploads."""
    return render_template(
        "index.html",
        error=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB. Please upload a smaller image."
    ), 413

@app.errorhandler(404)
def not_found(error):
    return render_template("index.html", error="Page not found (404)."), 404

if __name__ == "__main__":
    # Task 4.4: Run in Flask server with debug mode OFF
    DEBUG_MODE = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1")
    PORT = int(os.getenv("PORT", "5000"))
    print(f"[*] Starting VisionGuard Web App on http://0.0.0.0:{PORT} (debug={DEBUG_MODE})...")
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
