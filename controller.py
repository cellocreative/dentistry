''' File who coordinates what do with the inputs from the frontend. '''
import flask
from flask_cors import CORS
from model.model import update_model_weights
from model.prediction import load_model, predict_image
import random

# Initialize the Flask application and the model
app = flask.Flask(__name__)
CORS(app)
@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"

@app.route("/predict", methods=["POST"])
def predict():
    ''' Method to predict on an image. '''

    # Response to the frontend
    data = {"success": False}

    if flask.request.method == "POST":

        # Ensure an image was properly uploaded to our endpoint
        if flask.request.files.get("image"):

            # Convert the image
            image = flask.request.files["image"].read()

            probs, original, heatmap = predict_image(image)

            # Fill the response
            data["success"] = True
            data["id"] = random.randint(0, 100)
            data["probs"] = probs
            data["original"] = original
            data["heatmap"] = heatmap

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# Start the server
if __name__ == "__main__":
    print("* Loading the model and starting Flask server... " + \
            "please wait until server has fully started")

    update_model_weights()
    load_model()
    app.run()
