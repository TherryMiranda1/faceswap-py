import cv2
import io
import os
import logging
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from flask import Flask, request, jsonify, send_file, render_template

UPLOAD_FOLDER = 'uploads'

app_flask = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuración del modelo
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640))

# Carga del modelo inswapper
swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx',
                                          download=False,
                                          download_zip=False)

def allowed_file(filename):
    """Verificar si el archivo es una imagen permitida."""
    return '.' in filename and filename.lower().rsplit('.', 1)[1] in {'png', 'jpg', 'jpeg'}

def encode_image(image):
    """Codificar la imagen para enviar."""
    _, img_encoded = cv2.imencode('.jpg', image)
    return io.BytesIO(img_encoded.tobytes())

def format_face_to_image(face, image):
    """Formatear una cara a una imagen."""
    bbox = face['bbox']
    bbox = [int(b) for b in bbox]  
    img = cv2.imread(image)
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def clear_uploads_folder(upload_dir=UPLOAD_FOLDER):
    if os.path.exists(upload_dir):
        for file in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, file)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Archivo eliminado: {file_path}")
                except Exception as e:
                    logger.error(f"Error al eliminar {file_path}: {e}")


def save_uploaded_file(uploaded_file, upload_dir=UPLOAD_FOLDER):
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    # clear_uploads_folder(upload_dir)
    file_path = os.path.join(upload_dir, uploaded_file.filename)
    try:
        uploaded_file.save(file_path)
        logger.info(f"Archivo guardado: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error al guardar el archivo: {e}")
        raise e


@app_flask.route('/', methods=['GET'])
def visit():
    logger.info("App visitada")
    return render_template('welcome.html')

@app_flask.route('/health', methods=['GET'])
def health_check():
    logger.info("Endpoint de salud consultado.")
    return jsonify({"message": "La API está funcionando correctamente."})


@app_flask.route('/swap_faces/', methods=['POST'])
def swap_faces():
    try:
        
        # Obtener imágenes de la solicitud
        target_file = request.files.get('target')
        source_file = request.files.get('source_face')
        
        if not source_file or not target_file:
            return jsonify({"error": "Both source and target images are required"}), 400
        
        # Validar tipos de archivo
        if not (allowed_file(source_file.filename) and allowed_file(target_file.filename)):
            return jsonify({"error": "Invalid file type. Only .png, .jpg, .jpeg are allowed"}), 400
        

        source_image = save_uploaded_file(source_file)
        target_image = save_uploaded_file(target_file)
        logger.info(f"Archivos cargados: /uploads/{source_image} y /uploads/{target_image}")
       
        source_ins_image = cv2.imread(source_image)
        target_ins_image = cv2.imread(target_image)

        max_width = 640
        target_width = min(max_width, source_ins_image.shape[1])
        
        # Unificar las dimensiones de las imagenes
        aspect_ratio = target_ins_image.shape[1] / target_ins_image.shape[0]
        target_ins_image_resized = cv2.resize(target_ins_image, (target_width, int(max_width/ aspect_ratio)))
        
        # Obtener las caras de ambas imágenes
        source_faces = face_app.get(source_ins_image) 
        target_faces = face_app.get(target_ins_image_resized)  
        
        if len(source_faces) == 0:
            return jsonify({"error": "No face detected in source image"}), 400
        if len(target_faces) == 0:
            return jsonify({"error": "No face detected in target image"}), 400
        
        logger.info(f"Caras obtenidas: {len(source_faces)} y {len(target_faces)}")
            
        source_face = source_faces[0]
        target_face = target_faces[0] 
        
        # Realizar el intercambio de caras
        try:
            result_image = swapper.get(target_ins_image_resized, target_face, source_face, paste_back=True)
        except Exception as e:
            logger.error(f"Error al intercambiar caras: {e}")
            return jsonify({"error": "Failed to swap faces"}), 500
        
        # Codificar la imagen para enviar
        result_io = encode_image(result_image) 
        
        # Eliminar los archivos temporales
        clear_uploads_folder() 
        
        # Devolver la imagen procesada
        return send_file(result_io, mimetype='image/jpeg')
    
    except Exception as e:
        logger.error(f"Error desconocido: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app_flask.run(host='0.0.0.0')
