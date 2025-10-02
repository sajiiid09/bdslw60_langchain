import argparse
import os
import json
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import tempfile

from src.inference.predictor import SignLanguagePredictor, load_predictor_from_checkpoint


# HTML template for the demo interface
DEMO_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>BdSLW60 Sign Language Recognition Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .upload-area { 
            border: 2px dashed #ccc; 
            padding: 40px; 
            text-align: center; 
            margin: 20px 0;
            border-radius: 10px;
        }
        .result { 
            background: #f5f5f5; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 5px;
        }
        .error { background: #ffebee; color: #c62828; }
        .success { background: #e8f5e8; color: #2e7d32; }
        button { 
            background: #2196f3; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
        }
        button:hover { background: #1976d2; }
        .loading { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>BdSLW60 Sign Language Recognition Demo</h1>
        <p>Upload a video file to recognize the sign language gesture.</p>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area">
                <input type="file" id="videoFile" name="video" accept="video/*" required>
                <p>Select a video file (.mp4, .avi, .mov, .mkv)</p>
            </div>
            <button type="submit">Recognize Sign Language</button>
        </form>
        
        <div id="loading" class="loading">
            <p>Processing video... Please wait.</p>
        </div>
        
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('videoFile');
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');
            
            if (!fileInput.files[0]) {
                resultDiv.innerHTML = '<div class="result error">Please select a video file.</div>';
                return;
            }
            
            const formData = new FormData();
            formData.append('video', fileInput.files[0]);
            
            loadingDiv.style.display = 'block';
            resultDiv.innerHTML = '';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.error) {
                    resultDiv.innerHTML = `<div class="result error">Error: ${result.error}</div>`;
                } else {
                    const confidence = (result.confidence * 100).toFixed(1);
                    resultDiv.innerHTML = `
                        <div class="result success">
                            <h3>Recognition Result</h3>
                            <p><strong>Predicted Class:</strong> ${result.predicted_class}</p>
                            <p><strong>Confidence:</strong> ${confidence}%</p>
                            <p><strong>Top 5 Predictions:</strong></p>
                            <ul>
                                ${result.top_predictions.map(p => 
                                    `<li>Class ${p.class_id}: ${(p.probability * 100).toFixed(1)}%</li>`
                                ).join('')}
                            </ul>
                        </div>
                    `;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error">Error: ${error.message}</div>`;
            } finally {
                loadingDiv.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""


app = Flask(__name__)
predictor = None


def create_app(checkpoint_path: str, model_config_path: str, device: str = "auto"):
    """Create Flask app with predictor."""
    global predictor
    
    # Load predictor
    predictor = load_predictor_from_checkpoint(checkpoint_path, model_config_path, device)
    
    @app.route('/')
    def index():
        return render_template_string(DEMO_HTML)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            if 'video' not in request.files:
                return jsonify({'error': 'No video file provided'})
            
            file = request.files['video']
            if file.filename == '':
                return jsonify({'error': 'No video file selected'})
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                file.save(tmp_file.name)
                
                try:
                    # Make prediction
                    result = predictor.predict(tmp_file.name)
                    
                    # Get top 5 predictions
                    probabilities = result['probabilities']
                    top_indices = np.argsort(probabilities)[-5:][::-1]
                    top_predictions = [
                        {
                            'class_id': int(idx),
                            'probability': float(probabilities[idx])
                        }
                        for idx in top_indices
                    ]
                    
                    return jsonify({
                        'predicted_class': int(result['predicted_class']),
                        'confidence': float(result['confidence']),
                        'top_predictions': top_predictions
                    })
                    
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
                    
        except Exception as e:
            return jsonify({'error': str(e)})
    
    @app.route('/model_info')
    def model_info():
        return jsonify(predictor.get_model_info())
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Start demo server for sign language recognition")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model-config", required=True, help="Path to model config")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    
    # Create app
    app = create_app(args.checkpoint, args.model_config, args.device)
    
    print(f"Starting demo server...")
    print(f"Model info: {predictor.get_model_info()}")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    
    # Run app
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
