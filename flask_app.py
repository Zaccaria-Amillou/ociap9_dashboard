import os
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for

# Nombre des images
NB_IMAGES = 20

# definition des variables
selected_id = 1
img_paths = "static/data/img/"
mask_paths = "static/data/mask/"

# Catégories des images
cats = {
    "void": [0, 1, 2, 3, 4, 5, 6],
    "flat": [7, 8, 9, 10],
    "construction": [11, 12, 13, 14, 15, 16],
    "object": [17, 18, 19, 20],
    "nature": [21, 22],
    "sky": [23],
    "human": [24, 25],
    "vehicle": [26, 27, 28, 29, 30, 31, 32, 33, -1],
}

# Idéntifiant des catégories
cats_id = {
    "void": (0),
    "flat": (1),
    "construction": (2),
    "object": (3),
    "nature": (4),
    "sky": (5),
    "human": (6),
    "vehicle": (7),
}

# Couleurs des catégories
cats_colors = {
    0: (0, 0, 0),
    1: (50, 50, 50),
    2: (150, 150, 150),
    3: (255, 0, 0),
    4: (0, 255, 0),
    5: (0, 0, 255),
    6: (200, 200, 0),
    7: (150, 0, 200),
}


def get_data_prepared(path_X, dim):
    """Prépare les données pour la segmentation."""
    X = np.array([cv2.resize(cv2.cvtColor(cv2.imread(path_X), cv2.COLOR_BGR2RGB), dim)])
    X = X / 255

    return X


def prepare_img(img, dim):
    """Prépare l'image pour la segmentation."""
    X = np.array([cv2.resize(np.array(img), dim)])
    X = X / 255

    return X


# Recupère les chemins d'accès des fichiers
def getPathFiles():
    """Récupère les chemins d'accès des fichiers."""
    path_files = []

    # img set
    for file in os.listdir(img_paths):
        path_files.append(file.replace("leftImg8bit.png", ""))

    return path_files


path_files = getPathFiles()

app = Flask(__name__)

def load_model(model_choice):
    if model_choice == "Classic":
        model_path = "model/ResNet50_U-Net_augmented.tflite"
    elif model_choice == "New":
        model_path = "model/ResNet101_Coarse_U-Net_OCR.tflite"
    else:  # Assuming the third option is "Advanced"
        model_path = "model/ResNet101_U-Net_CAA_ augmented.tflite"
    model = tf.lite.Interpreter(model_path=model_path)
    model.allocate_tensors()
    return model, model_choice



@app.route("/", methods=["GET", "POST"])
def homepage():
    """Route pour la page d'accueil."""
    return render_template("index.html")


@app.route("/prediction", methods=["GET"])
@app.route("/prediction/", methods=["GET"])
def get_prediction():
    """Route pour obtenir une prédiction."""
    global selected_id
    if request.args.get("file"):
        selected_id = int(request.args.get("file"))

    img_path = img_paths + path_files[selected_id - 1] + "leftImg8bit.png"
    mask_path = mask_paths + path_files[selected_id - 1] + "gtFine_labelIds.png"

    img = cv2.resize(cv2.imread(img_path), (400, 200))
    mask = cv2.resize(cv2.imread(mask_path), (400, 200))
    mask = np.squeeze(mask[:, :, 0])
    mask_labelids = np.zeros((mask.shape[0], mask.shape[1], len(cats_id)))

    for i in range(-1, 34):
        for cat in cats:
            if i in cats[cat]:
                mask_labelids[:, :, cats_id[cat]] = np.logical_or(
                    mask_labelids[:, :, cats_id[cat]], (mask == i)
                )
                break

    mask_labelids = np.array(np.argmax(mask_labelids, axis=2), dtype="uint8")

    m = np.empty((mask_labelids.shape[0], mask_labelids.shape[1], 3), dtype="uint8")
    for i in range(mask_labelids.shape[0]):
        for j in range(mask_labelids.shape[1]):
            m[i][j] = cats_colors[mask_labelids[i][j]]

    cv2.imwrite("static/data/predict/img.png", img)
    cv2.imwrite("static/data/predict/mask.png", m)

    return render_template(
        "prediction.html", sended=False, nb_image=NB_IMAGES, selected=selected_id
    )


@app.route("/prediction", methods=["POST"])
@app.route("/prediction/", methods=["POST"])
def post_prediction():
    """Route pour poster une prédiction."""
    global selected_id
    if request.form.get("file"):
        selected_id = int(request.form.get("file"))

    return redirect(url_for("get_prediction", file=selected_id))


@app.route("/predict/", methods=["GET", "POST"])
def predictImage():
    """Route pour prédire une image avec trois modèles différents."""
    img_path = img_paths + path_files[selected_id - 1] + "leftImg8bit.png"

    # Load the original image and resize it
    original_img = cv2.imread(img_path)
    original_img_resized = cv2.resize(original_img, (400, 200))

    # Prepare the image once for all models
    img = get_data_prepared(img_path, (256, 256))
    img = img.astype("float32")

    # Load the three models
    models = {
        "Classic": load_model("Classic")[0],
        "New": load_model("New")[0],
        "Advanced": load_model("Advanced")[0]
    }

    for model_name, model in models.items():
        # Predict with each model
        input_details = model.get_input_details()
        model.set_tensor(input_details[0]["index"], img)
        model.invoke()
        y_pred = model.get_tensor(model.get_output_details()[0]["index"])
        y_pred_argmax = np.argmax(y_pred, axis=3)

        # Convert prediction to color image
        m = np.empty((y_pred_argmax[0].shape[0], y_pred_argmax[0].shape[1], 3), dtype="uint8")
        for i in range(y_pred_argmax[0].shape[0]):
            for j in range(y_pred_argmax[0].shape[1]):
                m[i][j] = cats_colors[y_pred_argmax[0][i][j]]

        # Resize for uniformity
        resized_m = cv2.resize(m, (400, 200))

        # Overlay the mask on the resized original image
        overlay = cv2.addWeighted(original_img_resized, 0.6, resized_m, 0.4, 0)

        # Save the prediction image and the overlay image for each model
        cv2.imwrite(f"static/data/predict/mask_predicted_{model_name.lower()}.png", resized_m)
        cv2.imwrite(f"static/data/predict/overlay_{model_name.lower()}.png", overlay)

    return render_template(
        "prediction.html", sended=True, nb_image=NB_IMAGES, selected=selected_id
    )



if __name__ == "__main__":
    app.run()
