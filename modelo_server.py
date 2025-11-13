#!/usr/bin/env python3
import os
import sys
import json
import logging
import time
import threading
from flask import Flask, request, jsonify
import numpy as np

# Configurar logging optimizado para Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MODEL_SERVER")

# SILENCIAR LOGS
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def inicializar_modelos():
    global analizador, modelos_listos, inicializacion_en_curso
    
    if inicializacion_en_curso:
        return
        
    inicializacion_en_curso = True
    logger.info("🔄 INICIANDO CARGA DE MODELOS EN RAILWAY...")
    
    try:
        # Agregar el directorio actual al path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(script_dir)
        
        logger.info("📁 Importando analisis_imagen.py...")
        
        # Intentar importar desde analisis_imagen.py
        try:
            from analisis_imagen import ImageAnalyzer
            logger.info("✅ analisis_imagen.py importado correctamente")
        except ImportError as e:
            logger.error(f"❌ Error importando analisis_imagen.py: {e}")
            # Intentar con analisis_imagen_completo.py como fallback
            try:
                from analisis_imagen_completo import ImageAnalyzer
                logger.info("✅ analisis_imagen_completo.py importado como fallback")
            except ImportError as e2:
                logger.error(f"❌ Error importando ambos scripts: {e2}")
                return
        
        logger.info("🎯 Creando instancia de ImageAnalyzer...")
        analizador = ImageAnalyzer()
        
        logger.info("📦 Cargando modelos (esto puede tomar 20-30 segundos)...")
        analizador.load_models()
        
        modelos_listos = getattr(analizador, 'cargado', False)
        
        if modelos_listos:
            logger.info("🎉 TODOS LOS MODELOS CARGADOS CORRECTAMENTE!")
            logger.info("🚀 Servidor listo para recibir peticiones")
        else:
            logger.error("💥 ERROR: No se pudieron cargar los modelos")
            
        inicializacion_en_curso = False
        
    except Exception as e:
        logger.error(f"💥 ERROR CRÍTICO en inicialización: {e}")
        import traceback
        logger.error(traceback.format_exc())
        modelos_listos = False
        inicializacion_en_curso = False

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ready" if modelos_listos else "initializing",
        "modelos_listos": modelos_listos,
        "inicializacion_en_curso": inicializacion_en_curso,
        "timestamp": time.time()
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if not modelos_listos:
        return jsonify({
            "error": "Modelos no listos",
            "es_apto": False,
            "puntuacion_riesgo": 1.0
        }), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data"}), 400
            
        image_path = data.get('image_path', '')
        image_url = data.get('image_url', '')
        
        if not image_path and not image_url:
            return jsonify({"error": "Se requiere image_path o image_url"}), 400
        
        # Para Railway, asumimos que las imágenes vienen por URL o path absoluto
        if image_url:
            # Aquí podrías implementar descarga de imagen desde URL
            return jsonify({"error": "URL analysis no implementado aún"}), 501
        
        # Verificar que la ruta existe
        if not os.path.exists(image_path):
            return jsonify({
                "error": f"Archivo no encontrado: {image_path}",
                "es_apto": False,
                "puntuacion_riesgo": 1.0
            }), 404

        logger.info(f"✅ Imagen encontrada, analizando: {image_path}")
        inicio = time.time()
        
        # Usar el método analyze_image del analizador
        if hasattr(analizador, 'analyze_image'):
            resultado = analizador.analyze_image(image_path)
        else:
            raise AttributeError("ImageAnalyzer no tiene método analyze_image")
        
        duracion = time.time() - inicio
        
        resultado["tiempo_procesamiento"] = duracion
        resultado["ruta_imagen"] = image_path
        
        logger.info(f"✅ Análisis completado en {duracion:.2f}s - Resultado: {'✅ APTO' if resultado.get('es_apto') else '❌ NO APTO'}")
        
        return jsonify(resultado)
        
    except Exception as e:
        logger.error(f"❌ Error en análisis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "es_apto": False,
            "puntuacion_riesgo": 1.0
        }), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "🚀 Servidor de Modelos de Moderación - Railway",
        "status": "running" if modelos_listos else "starting",
        "modelos_cargados": modelos_listos,
        "endpoints": {
            "GET /health": "Estado del servidor y modelos",
            "POST /analyze": "Analizar imagen (JSON: {image_path: 'ruta'})"
        }
    })

if __name__ == '__main__':
    # Inicializar modelos en segundo plano
    logger.info("🎯 Inicializando modelos en segundo plano...")
    thread = threading.Thread(target=inicializar_modelos, daemon=True)
    thread.start()
    
    port = int(os.environ.get('PORT', 5000))
    
    print("=" * 60)
    print("🚀 INICIANDO SERVIDOR DE MODELOS EN RAILWAY")
    print(f"🌐 Puerto: {port}")
    print("⏰ Hora de inicio:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=port,
        threaded=True,
        debug=False
    )