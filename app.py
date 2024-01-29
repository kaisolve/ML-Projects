from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

sign = { 0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
        11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
        21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')

@app.route("/")
def home():
  return "Working"

@app.route("/predict")
def predict():
  data = request.get_json()
  landmarks = data["landmarks"]

  predict = model.predict(np.array([landmarks]))
  label = np.argmax(np.squeeze(predict))

  return jsonify({ 'label' : sign[label] })

if __name__ == '__main__':
  app.run(debug=True)