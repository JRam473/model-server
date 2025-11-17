#//modelo_server.py
#!/usr/bin/env python3
import os
import sys
import json
import logging
import time
import threading
from flask import Flask, request, jsonify
import numpy as np

# ✅ CORREGIDO: Crear directorio logs antes de configurar logging
os.makedirs('logs', exist_ok=True)

# Configurar logging para Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/model_server.log', encoding='utf-8')  # ✅ Agregar encoding
    ]
)
logger = logging.getLogger("MODEL_SERVER_RENDER")

# SILENCIAR LOGS EXCESIVOS
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

app = Flask(__name__)

# Variables globales
analizador = None
modelos_listos = False
inicializacion_en_curso = False

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

def descargar_modelos_si_es_necesario():
    """Función para precargar modelos en el build de Render"""
    logger.info("🔍 Verificando modelos...")
    try:
        # Esto fuerza la descarga de modelos durante el build
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("✅ Modelo CLIP verificado")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo precargar modelos: {e}")

def inicializar_modelos():
    global analizador, modelos_listos, inicializacion_en_curso
    
    if inicializacion_en_curso:
        return
        
    inicializacion_en_curso = True
    logger.info("🔄 INICIANDO CARGA DE MODELOS EN RENDER...")
    
    try:
        # Agregar el directorio actual al path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(script_dir)
        
        # Intentar importar en orden de preferencia
        analizador = None
        
        for script_name in ['analisis_imagen', 'analisis_imagen_completo']:
            try:
                logger.info(f"📁 Intentando importar desde {script_name}.py...")
                module = __import__(script_name)
                if hasattr(module, 'ImageAnalyzer'):
                    analizador = module.ImageAnalyzer()
                    logger.info(f"✅ {script_name}.py importado correctamente")
                    break
            except ImportError as e:
                logger.warning(f"❌ Error importando {script_name}.py: {e}")
                continue
        
        if analizador is None:
            logger.error("💥 No se pudo importar ningún analizador")
            return
        
        logger.info("📦 Cargando modelos (esto puede tomar 1-2 minutos en Render)...")
        
        # Método de carga específico
        if hasattr(analizador, 'load_models'):
            analizador.load_models()
            modelos_listos = getattr(analizador, 'cargado', False)
        elif hasattr(analizador, 'load_model'):
            analizador.load_model()
            modelos_listos = True
        else:
            # Asumir que los modelos se cargan en el constructor
            modelos_listos = True
        
        if modelos_listos:
            logger.info("🎉 TODOS LOS MODELOS CARGADOS CORRECTAMENTE!")
            logger.info("🚀 Servidor listo para recibir peticiones")
        else:
            logger.error("💥 ERROR: No se pudieron cargar los modelos")
            
        inicializacion_en_curso = False
        
    except Exception as e:
        logger.error(f"💥 ERROR CRÍTICO en inicialización: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        modelos_listos = False
        inicializacion_en_curso = False

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de salud optimizado para Render"""
    health_status = {
        "status": "ready" if modelos_listos else "initializing",
        "modelos_listos": modelos_listos,
        "inicializacion_en_curso": inicializacion_en_curso,
        "timestamp": time.time(),
        "service": "tahitic-model-server"
    }
    
    status_code = 200 if modelos_listos else 503
    return jsonify(health_status), status_code

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Endpoint principal de análisis"""
    if not modelos_listos:
        return jsonify({
            "error": "Modelos no listos, intente más tarde",
            "es_apto": False,
            "puntuacion_riesgo": 1.0
        }), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se recibieron datos JSON"}), 400
            
        image_path = data.get('image_path', '')
        
        if not image_path:
            return jsonify({"error": "Se requiere image_path"}), 400
        
        # En Render, las imágenes deben estar accesibles via URL o path absoluto
        if not os.path.exists(image_path):
            return jsonify({
                "error": f"Archivo no encontrado: {image_path}",
                "sugerencia": "En Render, use URLs accesibles públicamente",
                "es_apto": False,
                "puntuacion_riesgo": 1.0
            }), 404

        logger.info(f"🔍 Analizando imagen: {image_path}")
        inicio = time.time()
        
        # Usar el método analyze_image del analizador
        if hasattr(analizador, 'analyze_image'):
            resultado = analizador.analyze_image(image_path)
        else:
            raise AttributeError("El analizador no tiene método analyze_image")
        
        duracion = time.time() - inicio
        
        # Formatear respuesta
        respuesta = {
            "es_apto": resultado.get('es_apto', False),
            "puntuacion_riesgo": resultado.get('puntuacion_riesgo', 1.0),
            "tiempo_procesamiento": duracion,
            "analizado_en": time.strftime("%Y-%m-%d %H:%M:%S"),
            "servidor": "render-model-server"
        }
        
        # Incluir detalles si existen
        if 'analisis_violencia' in resultado:
            respuesta['analisis_violencia'] = resultado['analisis_violencia']
        if 'analisis_armas' in resultado:
            respuesta['analisis_armas'] = resultado['analisis_armas']
        
        logger.info(f"✅ Análisis completado en {duracion:.2f}s - Apto: {respuesta['es_apto']}")
        
        return jsonify(respuesta)
        
    except Exception as e:
        logger.error(f"❌ Error en análisis: {e}")
        return jsonify({
            "error": f"Error procesando imagen: {str(e)}",
            "es_apto": False,
            "puntuacion_riesgo": 1.0
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """Endpoint extendido de estado"""
    return jsonify({
        "service": "Tahitic Model Server",
        "status": "operational" if modelos_listos else "initializing",
        "modelos_cargados": modelos_listos,
        "timestamp": time.time(),
        "environment": "production",
        "version": "1.0.0"
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "🚀 Servidor de Modelos de Moderación - Render",
        "status": "running" if modelos_listos else "starting",
        "documentation": {
            "GET /health": "Estado de salud del servidor",
            "POST /analyze": "Analizar imagen (JSON: {image_path: 'ruta'})",
            "GET /status": "Estado extendido del servicio"
        }
    })

# Inicialización al importar
logger.info("🎯 Preparando inicialización de modelos...")

# Iniciar carga de modelos en segundo plano inmediatamente
modelos_thread = threading.Thread(target=inicializar_modelos, daemon=True)
modelos_thread.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 50000))
    
    print("=" * 60)
    print("🚀 INICIANDO SERVIDOR DE MODELOS EN RENDER")
    print(f"🌐 Puerto: {port}")
    print(f"📁 Directorio: {os.getcwd()}")
    print("🐍 Python:", sys.version)
    print("⏰ Hora de inicio:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # En Render, usar gunicorn es mejor para producción
    if os.environ.get('RENDER'):
        # Esto se ejecutará en Render
        app.run(host='0.0.0.0', port=port, threaded=True)
    else:
        # Para desarrollo local
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)