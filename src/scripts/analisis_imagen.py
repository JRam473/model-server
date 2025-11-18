#!/usr/bin/env python3
import sys
import json
import logging
import os
from PIL import Image
import numpy as np
import gc

# ✅ CONFIGURACIÓN MÍNIMA PARA PRODUCCIÓN
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MODERACION_OPTIMIZADA")

# ✅ SILENCIAR COMPLETAMENTE
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

class WeaponDetectorOptimizado:
    def __init__(self):
        self.model = None
        self.cargado = False
        self.model_name = "YOLOv8n"
        self.model_type = None

    def load_model(self):
        """Carga optimizada de YOLO - versión ligera"""
        try:
            from ultralytics import YOLO
            
            # ✅ YOLOv8n es el más pequeño
            self.model = YOLO('yolov8n.pt')
            self.model_type = 'yolo'
            self.cargado = True
            
            # ✅ WARM-UP RÁPIDO CON IMAGEN PEQUEÑA
            try:
                dummy = np.ones((320, 320, 3), dtype=np.uint8)
                _ = self.model(dummy, verbose=False)
            except:
                pass  # Ignorar errores en warm-up
                
            logger.info("✅ YOLOv8n cargado (modo optimizado)")
            
        except ImportError as e:
            logger.error(f"❌ YOLO no disponible: {e}")
            self.cargado = False
        except Exception as e:
            logger.error(f"❌ Error cargando YOLO: {e}")
            self.cargado = False

    def analyze_weapons(self, image_path: str):
        """Análisis optimizado de armas"""
        if not self.cargado:
            return {"armas_detectadas": False, "confianza": 0.0, "error": "Modelo no cargado"}

        try:
            # ✅ CONFIGURACIÓN OPTIMIZADA - MENOR RESOLUCIÓN
            results = self.model(image_path, verbose=False, conf=0.3, imgsz=640)
            weapons_detected = []
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # ✅ SOLO CATEGORÍAS ESENCIALES
                    if class_name in ['knife', 'gun'] and confidence > 0.3:
                        weapons_detected.append({
                            'weapon': class_name,
                            'confidence': confidence
                        })
            
            armas_detectadas = len(weapons_detected) > 0
            confianza_max = max([w['confidence'] for w in weapons_detected]) if weapons_detected else 0.0
            
            return {
                "armas_detectadas": armas_detectadas,
                "confianza": confianza_max,
                "detalles_armas": weapons_detected,
                "total_armas_detectadas": len(weapons_detected),
                "modelo_utilizado": "YOLOv8n-optimizado"
            }
            
        except Exception as e:
            logger.error(f"❌ Error analizando armas: {e}")
            return {"armas_detectadas": False, "confianza": 0.0, "error": str(e)}

class ViolenceDetectorOptimizado:
    def __init__(self):
        self.model = None
        self.cargado = False
        self.model_name = "CLIP-base"

    def load_model(self):
        """Carga optimizada de CLIP - versión base más pequeña"""
        try:
            from transformers import pipeline
            
            # ✅ CLIP base es más pequeño que large
            self.classifier = pipeline(
                "zero-shot-image-classification",
                model="openai/clip-vit-base-patch32",
                device=-1  # ✅ FORZAR CPU
            )
            self.cargado = True
            
            # ✅ CACHE DE ETIQUETAS OPTIMIZADAS
            self.candidate_labels = [
                "violence", "weapon", "blood", "nudity", "sexual content",
                "safe content", "normal scene", "peaceful image"
            ]
            
            logger.info("✅ CLIP base cargado (modo optimizado)")
            
        except Exception as e:
            logger.error(f"❌ Error cargando CLIP: {e}")
            self.cargado = False

    def analyze_violence(self, image_path: str):
        """Análisis optimizado de violencia"""
        if not self.cargado:
            return {"es_violento": False, "probabilidad_violencia": 0.0, "error": "Modelo no cargado"}

        try:
            result = self.classifier(image_path, candidate_labels=self.candidate_labels)
            violencia_detectada = []
            
            for pred in result:
                score = pred['score']
                label_lower = pred['label'].lower()
                
                # ✅ UMBRAL MÁS ALTO PARA REDUCIR FALSOS POSITIVOS
                if score > 0.3:
                    # ✅ SOLO CATEGORÍAS DE ALTA PRIORIDAD
                    if any(keyword in label_lower for keyword in ['violence', 'weapon', 'blood', 'nudity', 'sexual']):
                        violencia_detectada.append({
                            'label': pred['label'],
                            'score': pred['score'],
                            'tipo': 'alta_prioridad'
                        })
            
            es_violento = len(violencia_detectada) > 0
            probabilidad_violencia = max([v['score'] for v in violencia_detectada]) if violencia_detectada else 0.0
            
            return {
                "es_violento": es_violento,
                "probabilidad_violencia": probabilidad_violencia,
                "detalles_violencia": violencia_detectada,
                "total_categorias_analizadas": len(self.candidate_labels)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analizando violencia: {e}")
            return {"es_violento": False, "probabilidad_violencia": 0.0, "error": str(e)}

class ImageAnalyzerOptimizado:
    """
    ✅ VERSIÓN OPTIMIZADA PARA PRODUCCIÓN
    - Menor uso de memoria
    - Modelos más ligeros
    - Procesamiento más rápido
    """
    def __init__(self):
        self.weapon_detector = WeaponDetectorOptimizado()
        self.violence_detector = ViolenceDetectorOptimizado()
        self.cargado = False

    def load_models(self):
        """Carga optimizada de modelos"""
        logger.info("🔄 Cargando modelos optimizados...")
        try:
            # ✅ CARGA SECUENCIAL PARA EVITAR PICO DE MEMORIA
            self.violence_detector.load_model()
            
            # ✅ PEQUEÑA PAUSA ENTRE CARGA DE MODELOS
            import time
            time.sleep(1)
            
            self.weapon_detector.load_model()
            
            self.cargado = self.weapon_detector.cargado and self.violence_detector.cargado
            
            if self.cargado:
                logger.info("✅ Todos los modelos cargados (modo optimizado)")
            else:
                logger.error("❌ Falló la carga de algún modelo")
                
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            self.cargado = False

    def preprocess_image(self, image_path: str, max_size: int = 512):
        """Preprocesamiento optimizado de imágenes"""
        try:
            img = Image.open(image_path)
            
            # ✅ REDUCIR TAMAÑO SI ES MUY GRANDE
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # ✅ GUARDAR TEMPORALMENTE
                import tempfile
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"opt_{os.path.basename(image_path)}")
                img.save(temp_path, optimize=True, quality=85)
                return temp_path
            
            return image_path
        except Exception as e:
            logger.warning(f"⚠️ Error en preprocesamiento, usando imagen original: {e}")
            return image_path

    def analyze_image(self, image_path: str):
        """Análisis completo optimizado"""
        if not self.cargado:
            return {"es_apto": False, "error": "Modelos no cargados", "puntuacion_riesgo": 1.0}

        temp_path = None
        try:
            if not os.path.exists(image_path):
                return {"es_apto": False, "error": "Archivo no encontrado", "puntuacion_riesgo": 1.0}

            # ✅ PREPROCESAR IMAGEN
            optimized_path = self.preprocess_image(image_path)
            temp_path = optimized_path if optimized_path != image_path else None

            # ✅ ANÁLISIS PARALELO OPTIMIZADO
            resultado_violencia = self.violence_detector.analyze_violence(optimized_path)
            resultado_armas = self.weapon_detector.analyze_weapons(optimized_path)
            
            # ✅ CÁLCULO OPTIMIZADO DE RIESGO
            riesgo_violencia = resultado_violencia.get("probabilidad_violencia", 0)
            riesgo_armas = resultado_armas.get("confianza", 0) if resultado_armas.get("armas_detectadas") else 0
            
            # ✅ POLÍTICA DE DECISIÓN MÁS EFICIENTE
            es_apto = not (
                (resultado_violencia.get("es_violento", False) and riesgo_violencia > 0.5) or
                (resultado_armas.get("armas_detectadas", False) and riesgo_armas > 0.4)
            )
            
            puntuacion_riesgo = max(riesgo_violencia, riesgo_armas)
            
            resultado_final = {
                "es_apto": es_apto,
                "analisis_violencia": resultado_violencia,
                "analisis_armas": resultado_armas,
                "puntuacion_riesgo": float(puntuacion_riesgo),
                "version": "optimizada-1.0"
            }
            
            return resultado_final

        except Exception as e:
            logger.error(f"❌ Error analizando imagen: {e}")
            return {"es_apto": False, "error": str(e), "puntuacion_riesgo": 1.0}
        
        finally:
            # ✅ LIMPIAR ARCHIVO TEMPORAL SI EXISTE
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            # ✅ LIMPIAR MEMORIA
            gc.collect()

# ✅ MANTENER COMPATIBILIDAD CON VERSIÓN ANTERIOR
class ImageAnalyzer(ImageAnalyzerOptimizado):
    """
    Clase legacy para mantener compatibilidad
    Hereda de la versión optimizada
    """
    def __init__(self):
        super().__init__()
        logger.info("🔁 Usando ImageAnalyzer (compatibilidad legacy)")

def main():
    """Función principal para uso standalone"""
    if len(sys.argv) != 2:
        error_msg = {"error": "Uso: analisis_imagen.py <ruta_imagen>"}
        print(json.dumps(error_msg))
        sys.exit(1)

    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        error_msg = {"error": f"Archivo no encontrado: {image_path}", "es_apto": False}
        print(json.dumps(error_msg))
        sys.exit(1)
    
    try:
        analyzer = ImageAnalyzerOptimizado()
        analyzer.load_models()
        
        if not analyzer.cargado:
            error_result = {"es_apto": False, "error": "Modelos no cargados", "puntuacion_riesgo": 1.0}
            print(json.dumps(error_result, ensure_ascii=False))
            sys.exit(1)
            
        result = analyzer.analyze_image(image_path)
        print(json.dumps(result, cls=CustomJSONEncoder, ensure_ascii=False, indent=2))
        
    except Exception as e:
        error_result = {"es_apto": False, "error": f"Error critico: {str(e)}", "puntuacion_riesgo": 1.0}
        print(json.dumps(error_result, ensure_ascii=False))

if __name__ == "__main__":
    main()