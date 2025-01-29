from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
import os
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'datasets/uploaded_dataset'
app.config['RESULT_FOLDER'] = 'static'

@app.route('/')
def index():
    return render_template('index.html', result_csv=None, error_message=None)

@app.route('/absa', methods=['POST'])
def upload_file():
    text_data = request.form.get('text')
    csv_file = request.files.get('file')
    result_csv = None
    error_message = None

    # If text data is provided, process it
    if text_data:
        try:
            result_csv = subprocess.check_output(
                ["python", "testing.py", text_data], text=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            print("Error:", e.output)  # Log the error output from main.py
            result_csv = None

        return render_template('index.html', text_data=text_data, result_csv=result_csv, error_message=error_message)

    # If a CSV file is uploaded, check if it's a valid CSV file
    elif csv_file and csv_file.filename != "":
        # Check if the uploaded file is a CSV by checking its extension
        if not csv_file.filename.lower().endswith('.csv'):
            error_message = "Unsupported file type. Please upload a CSV file."
            return render_template('index.html', result_csv=result_csv, error_message=error_message)

        filename = os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename)
        csv_file.save(filename)

        try:
            subprocess.run(["python", "testing.py", filename], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing file {filename}: {e}")

        return redirect(url_for('result'))

    # If no valid input is provided, redirect to the home page
    else:
        return redirect(url_for('index'))

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/datasets/uploaded_dataset/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
