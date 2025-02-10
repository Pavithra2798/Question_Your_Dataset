from flask import Flask, render_template, request,jsonify
import pandas as pd  
import analyze_dataset_qds

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index_qds.html')

@app.route('/submit', methods=['POST'])
def submit():    
    file = request.files['excelFile']
    question = request.json.get('question')
    if file:
        query, data, pattern, predictions = analyze_dataset_qds.get_excel(file, question)
        return jsonify({'video_summary': data})
    else:
        return "File not found"

@app.route('/process_video', methods=['POST'])
def process_video():
    recording_video = request.files['videoFile']
    question = request.form['question']

    file_path = '/tmp/recording_video.xlsx'

    recording_video.save(file_path)
    query, data, pattern, predictions = analyze_dataset_qds.get_excel(file_path, question)

    return jsonify({'tblData': [data.to_html(classes='data')],
    'patterns':pattern,
    'prediction':query  
    })

if __name__ == '__main__':
    # app.run()
    app.run(debug=True, port=8080)