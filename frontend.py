from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def root():
	return app.send_static_file('index.html')