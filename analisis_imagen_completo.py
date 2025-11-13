#!/usr/bin/env python3
import sys
import json
import logging
import os
from PIL import Image
import numpy as np

# Configurar logging MÍNIMO
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("moderacion_completa.log", encoding='utf-8'),
    ]
)
logger = logging.getLogger("MODERACION_COMPLETA")

# SILENCIAR YOLO
os.environ['YOLO_VERBOSE'] = 'False'

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

class WeaponDetector:
    def __init__(self):
        self.model = None
        self.cargado = False
        self.model_name = "YOLOv8n"
        self.model_type = None

    def load_model(self):
        """Carga el mejor modelo disponible para detección de armas"""
        try:
            # Intentar cargar YOLO primero
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')
                self.model_type = 'yolo'
                self.cargado = True
                logger.info("YOLOv8 cargado correctamente")
            except ImportError:
                from transformers import pipeline
                self.classifier = pipeline(
                    "zero-shot-image-classification", 
                    model="openai/clip-vit-base-patch32"
                )
                self.model_type = 'clip'
                self.model_name = "CLIP"
                self.cargado = True
                logger.info("CLIP cargado como fallback para armas")
        except Exception as e:
            logger.error(f"Error cargando modelo de armas: {e}")
            self.cargado = False

    def analyze_weapons(self, image_path: str):
        """Detección de armas con modelo ESPECIALIZADO"""
        if not self.cargado:
            return {"armas_detectadas": False, "confianza": 0.0, "error": "Modelo no cargado"}

        try:
            if self.model_type == 'yolo':
                results = self.model(image_path, verbose=False)
                weapons_detected = []
                
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        confidence = float(box.conf[0])
                        
                        if class_name in ['knife', 'gun'] and confidence > 0.5:
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
                    "modelo_utilizado": "YOLOv8"
                }
                
            else:
                candidate_labels = ["gun", "knife", "weapon", "firearm", "pistol", "rifle"]
                result = self.classifier(image_path, candidate_labels=candidate_labels)
                weapons_detected = [pred for pred in result if pred['score'] > 0.4]
                armas_detectadas = len(weapons_detected) > 0
                confianza_max = max([pred['score'] for pred in weapons_detected]) if weapons_detected else 0.0
                
                return {
                    "armas_detectadas": armas_detectadas,
                    "confianza": confianza_max,
                    "detalles_armas": weapons_detected,
                    "total_armas_detectadas": len(weapons_detected),
                    "modelo_utilizado": "CLIP"
                }
            
        except Exception as e:
            logger.error(f"Error analizando armas: {e}")
            return {"armas_detectadas": False, "confianza": 0.0, "error": str(e)}

class ViolenceDetector:
    def __init__(self):
        self.model = None
        self.cargado = False
        self.model_name = "CLIP"

    def load_model(self):
        """Carga modelo para detección de violencia"""
        try:
            from transformers import pipeline
            self.classifier = pipeline(
                "zero-shot-image-classification",
                model="openai/clip-vit-base-patch32"
            )
            self.cargado = True
            logger.info("Modelo CLIP cargado correctamente")
        except Exception as e:
            logger.error(f"Error cargando modelo CLIP: {e}")
            self.cargado = False

    def analyze_violence(self, image_path: str):
        """Analiza contenido violento"""
        if not self.cargado:
            return {"es_violento": False, "probabilidad_violencia": 0.0, "error": "Modelo no cargado"}

        try:
            candidate_labels = [
                "blood", "gore", "injured person", "bloody scene", "violence",
                "gun", "knife", "weapon", "firearm", "pistol", "rifle",
                "porn", "sexual content", "explicit content", "nudity",
                "fight", "battle", "war", "horror", "terror",
                "landscape", "peaceful image", "normal scene", "safe content",
                "person smiling", "everyday life", "nature", "building"
            ]
            
            result = self.classifier(image_path, candidate_labels=candidate_labels)
            violencia_detectada = []
            
            for pred in result:
                score = pred['score']
                label_lower = pred['label'].lower()
                
                if score > 0.15:
                    alta_prioridad = ['blood', 'gore', 'injured', 'bloody', 'porn', 'sexual', 'explicit', 'nudity', 'gun', 'knife', 'weapon', 'firearm', 'pistol', 'rifle']
                    if any(keyword in label_lower for keyword in alta_prioridad) and score > 0.1:
                        if any(arma in label_lower for arma in ['gun', 'knife', 'weapon', 'firearm', 'pistol', 'rifle']):
                            tipo = 'armas'
                        elif any(explicito in label_lower for explicito in ['porn', 'sexual', 'explicit', 'nudity']):
                            tipo = 'contenido_sexual'
                        else:
                            tipo = 'violencia_graphica'
                            
                        violencia_detectada.append({
                            'label': pred['label'],
                            'score': pred['score'],
                            'tipo': tipo,
                            'prioridad': 'alta'
                        })
                    
                    baja_prioridad = ['fight', 'battle', 'war', 'horror', 'terror', 'violence']
                    if any(keyword in label_lower for keyword in baja_prioridad) and score > 0.3:
                        violencia_detectada.append({
                            'label': pred['label'],
                            'score': pred['score'],
                            'tipo': 'violencia_general',
                            'prioridad': 'baja'
                        })
            
            es_violento = len(violencia_detectada) > 0
            probabilidad_violencia = max([v['score'] for v in violencia_detectada]) if violencia_detectada else 0.0
            
            return {
                "es_violento": es_violento,
                "probabilidad_violencia": probabilidad_violencia,
                "detalles_violencia": violencia_detectada,
                "total_categorias_analizadas": len(candidate_labels),
                "categorias_encontradas": [v['label'] for v in violencia_detectada]
            }
            
        except Exception as e:
            logger.error(f"Error analizando violencia: {e}")
            return {"es_violento": False, "probabilidad_violencia": 0.0, "error": str(e)}

class ImageAnalyzer:
    def __init__(self):
        self.weapon_detector = WeaponDetector()
        self.violence_detector = ViolenceDetector()
        self.cargado = False
        self.load_models()

    def load_models(self):
        """Carga todos los modelos necesarios"""
        try:
            self.violence_detector.load_model()
            self.weapon_detector.load_model()
            self.cargado = self.weapon_detector.cargado and self.violence_detector.cargado
            
            if self.cargado:
                logger.info("Todos los modelos cargados correctamente")
            else:
                logger.error("Falló la carga de algún modelo")
                
        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")
            self.cargado = False

    # ✅ MÉTODO FALTANTE AGREGADO
    def analyze_image(self, image_path: str):
        """Analiza una imagen para contenido inapropiado"""
        if not self.cargado:
            return {"es_apto": False, "error": "Modelos no cargados", "puntuacion_riesgo": 1.0}

        try:
            if not os.path.exists(image_path):
                return {"es_apto": False, "error": "Archivo no encontrado", "puntuacion_riesgo": 1.0}

            # Análisis paralelo
            resultado_violencia = self.violence_detector.analyze_violence(image_path)
            resultado_armas = self.weapon_detector.analyze_weapons(image_path)
            
            # Calcular riesgos
            riesgo_violencia = resultado_violencia.get("probabilidad_violencia", 0)
            riesgo_armas = resultado_armas.get("confianza", 0) if resultado_armas.get("armas_detectadas") else 0
            
            # Detectar armas en análisis de violencia
            armas_en_violencia = False
            confianza_armas_violencia = 0.0
            
            if resultado_violencia.get("detalles_violencia"):
                for deteccion in resultado_violencia["detalles_violencia"]:
                    if deteccion.get("tipo") == "armas":
                        armas_en_violencia = True
                        confianza_armas_violencia = max(confianza_armas_violencia, deteccion["score"])

            # ✅ CONFIGURACIÓN MÁS PERMISIVA
            es_apto = not (
                (resultado_violencia.get("es_violento", False) and riesgo_violencia > 0.7) or
                (resultado_armas.get("armas_detectadas", False) and riesgo_armas > 0.6) or
                (armas_en_violencia and confianza_armas_violencia > 0.5)
            )
            
            puntuacion_riesgo = max(riesgo_violencia, riesgo_armas)
            
            resultado_final = {
                "es_apto": es_apto,
                "analisis_violencia": resultado_violencia,
                "analisis_armas": resultado_armas,
                "puntuacion_riesgo": float(puntuacion_riesgo),
                "armas_detectadas_en_violencia": armas_en_violencia,
                "confianza_armas_violencia": float(confianza_armas_violencia)
            }
            
            return resultado_final

        except Exception as e:
            logger.error(f"Error analizando imagen: {e}")
            return {"es_apto": False, "error": str(e), "puntuacion_riesgo": 1.0}

def main():
    if len(sys.argv) != 2:
        error_msg = {"error": "Uso: moderacion_completa.py <ruta_imagen>"}
        print(json.dumps(error_msg))
        sys.exit(1)

    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        error_msg = {"error": f"Archivo no encontrado: {image_path}", "es_apto": False}
        print(json.dumps(error_msg))
        sys.exit(1)
    
    try:
        analyzer = ImageAnalyzer()
        result = analyzer.analyze_image(image_path)
        print(json.dumps(result, cls=CustomJSONEncoder, ensure_ascii=False, indent=2))
        
    except Exception as e:
        error_result = {"es_apto": False, "error": f"Error critico: {str(e)}", "puntuacion_riesgo": 1.0}
        print(json.dumps(error_result, ensure_ascii=False))

if __name__ == "__main__":
    main()