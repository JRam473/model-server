#!/usr/bin/env python3
import sys
import json
import logging
import os
from PIL import Image
import numpy as np

# Configurar logging COMPLETO
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
                
                logger.info("Cargando YOLOv8 para deteccion de armas...")
                
                self.model = YOLO('yolov8n.pt')
                self.model_type = 'yolo'
                self.model_name = "YOLOv8n"
                self.cargado = True
                
                logger.info("YOLOv8 cargado correctamente")
                logger.info("   - Tipo: Object detection")
                logger.info("   - Clases: 80 categorias incluyendo armas")
                
            except ImportError:
                logger.warning("YOLO no disponible, usando CLIP como fallback")
                from transformers import pipeline
                self.classifier = pipeline(
                    "zero-shot-image-classification", 
                    model="openai/clip-vit-base-patch32"
                )
                self.model_type = 'clip'
                self.model_name = "CLIP (fallback)"
                self.cargado = True
                logger.info("CLIP cargado como fallback para armas")
            
        except Exception as e:
            logger.error(f"ERROR CARGANDO MODELO DE ARMAS: {e}")
            self.cargado = False

    def analyze_weapons(self, image_path: str):
        """Detección de armas con modelo ESPECIALIZADO"""
        if not self.cargado:
            return {"armas_detectadas": False, "confianza": 0.0, "error": "Modelo no cargado"}

        try:
            logger.info(f"Analizando armas en: {image_path}")
            
            if self.model_type == 'yolo':
                # ✅ MEJORAR CONFIGURACIÓN YOLO - CONFIANZA MÁS BAJA
                results = self.model(image_path, verbose=False, conf=0.25)
                weapons_detected = []
                
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # ✅ MÁS CATEGORÍAS DE ARMAS Y OBJETOS PELIGROSOS
                        weapon_categories = [
                            'knife', 'gun', 'pistol', 'rifle', 'firearm', 'weapon',
                            'sword', 'dagger', 'machete', 'shotgun', 'revolver',
                            'scissors', 'axe', 'bat', 'hammer'
                        ]
                        
                        # ✅ UMBRAL MÁS BAJO PARA DETECCIÓN
                        if class_name in weapon_categories and confidence > 0.25:
                            weapons_detected.append({
                                'weapon': class_name,
                                'confidence': confidence
                            })
                            logger.info(f"   Detectado: {class_name} (confianza: {confidence:.4f})")
                
                armas_detectadas = len(weapons_detected) > 0
                confianza_max = max([w['confidence'] for w in weapons_detected]) if weapons_detected else 0.0
                
                resultado = {
                    "armas_detectadas": armas_detectadas,
                    "confianza": confianza_max,
                    "detalles_armas": weapons_detected,
                    "total_armas_detectadas": len(weapons_detected),
                    "modelo_utilizado": "YOLOv8"
                }
                
            else:
                # Detección con CLIP (FALLBACK)
                candidate_labels = [
                    "gun", "knife", "weapon", "firearm", "pistol", "rifle", 
                    "sword", "dagger", "machete", "shotgun", "revolver"
                ]
                
                logger.info(f"Buscando {len(candidate_labels)} tipos de armas...")
                result = self.classifier(image_path, candidate_labels=candidate_labels)
                
                # Log de predicciones de armas
                logger.info("PREDICCIONES DE ARMAS:")
                for i, pred in enumerate(result):
                    if pred['score'] > 0.1:
                        logger.info(f"   {i+1:2d}. {pred['label']:15s} : {pred['score']:.4f}")
                
                # ✅ UMBRAL MÁS BAJO PARA CLIP
                weapons_detected = [pred for pred in result if pred['score'] > 0.2]
                armas_detectadas = len(weapons_detected) > 0
                confianza_max = max([pred['score'] for pred in weapons_detected]) if weapons_detected else 0.0
                
                resultado = {
                    "armas_detectadas": armas_detectadas,
                    "confianza": confianza_max,
                    "detalles_armas": weapons_detected,
                    "total_armas_detectadas": len(weapons_detected),
                    "modelo_utilizado": "CLIP"
                }
            
            # Log del resultado
            if armas_detectadas:
                logger.warning(f"ARMAS DETECTADAS: {resultado['total_armas_detectadas']} armas encontradas")
                for arma in resultado['detalles_armas']:
                    logger.warning(f"   - {arma.get('weapon', arma.get('label', 'arma'))} (confianza: {arma.get('confidence', arma.get('score', 0)):.4f})")
            else:
                logger.info("No se detectaron armas")
            
            logger.info(f"RESULTADO ARMAS: detectadas={armas_detectadas}, confianza_max={confianza_max:.4f}")
            return resultado
            
        except Exception as e:
            logger.error(f"ERROR ANALIZANDO ARMAS: {e}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {"armas_detectadas": False, "confianza": 0.0, "error": str(e)}

class ViolenceDetector:
    def __init__(self):
        self.model = None
        self.cargado = False
        self.model_name = "CLIP (Zero-Shot)"

    def load_model(self):
        """Carga modelo ESPECIALIZADO para detección de violencia"""
        try:
            from transformers import pipeline
            
            logger.info("Cargando modelo CLIP para clasificacion flexible...")
            
            self.classifier = pipeline(
                "zero-shot-image-classification",
                model="openai/clip-vit-base-patch32"
            )
            self.cargado = True
            self.model_name = "openai/clip-vit-base-patch32"
            logger.info("Modelo CLIP cargado correctamente")
            logger.info("   - Tipo: Zero-shot image classification")
            logger.info("   - Capacidad: Clasificacion flexible con categorias personalizadas")
            
        except Exception as e:
            logger.error(f"Error cargando modelo CLIP: {e}")
            self.cargado = False

    def analyze_violence(self, image_path: str):
        """Analiza contenido violento con modelo ESPECIALIZADO"""
        if not self.cargado:
            return {
                "es_violento": False, 
                "probabilidad_violencia": 0.0, 
                "error": "Modelo no cargado"
            }

        try:
            logger.info(f"Analizando violencia en: {image_path}")
            
            # Categorías para violencia
            candidate_labels = [
                # CONTENIDO EXPLÍCITO Y ARMAS (ALTA PRIORIDAD)
                "blood", "gore", "injured person", "bloody scene", "violence",
                "gun", "knife", "weapon", "firearm", "pistol", "rifle",
                "porn", "sexual content", "explicit content", "nudity",
                
                # SITUACIONES PELIGROSAS (BAJA PRIORIDAD)
                "fight", "battle", "war", "horror", "terror",
                
                # CONTENIDO SEGURO (para contraste)
                "landscape", "peaceful image", "normal scene", "safe content",
                "person smiling", "everyday life", "nature", "building"
            ]
            
            logger.info(f"Buscando {len(candidate_labels)} categorias...")
            
            # Ejecutar clasificación
            result = self.classifier(image_path, candidate_labels=candidate_labels)
            
            # Log de todas las predicciones
            logger.info("PREDICCIONES DE VIOLENCIA (Top 10):")
            for i, pred in enumerate(result[:10]):
                logger.info(f"   {i+1:2d}. {pred['label']:25s} : {pred['score']:.4f}")
            
            # Filtrar predicciones de violencia
            violencia_detectada = []
            
            for pred in result:
                score = pred['score']
                label_lower = pred['label'].lower()
                
                if score > 0.15:
                    # CATEGORÍAS DE ALTA PRIORIDAD
                    alta_prioridad = [
                        'blood', 'gore', 'injured', 'bloody', 'porn', 'sexual', 'explicit', 'nudity',
                        'gun', 'knife', 'weapon', 'firearm', 'pistol', 'rifle'
                    ]
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
                    
                    # CATEGORÍAS DE BAJA PRIORIDAD
                    baja_prioridad = ['fight', 'battle', 'war', 'horror', 'terror', 'violence']
                    if any(keyword in label_lower for keyword in baja_prioridad) and score > 0.3:
                        violencia_detectada.append({
                            'label': pred['label'],
                            'score': pred['score'],
                            'tipo': 'violencia_general',
                            'prioridad': 'baja'
                        })
            
            # Determinar resultado
            es_violento = len(violencia_detectada) > 0
            probabilidad_violencia = max([v['score'] for v in violencia_detectada]) if violencia_detectada else 0.0
            
            # Log de detecciones específicas
            if violencia_detectada:
                logger.warning("DETECCIONES DE VIOLENCIA ENCONTRADAS:")
                for deteccion in violencia_detectada:
                    logger.warning(f"   - {deteccion['label']} ({deteccion['score']:.4f}) - Tipo: {deteccion['tipo']} - Prioridad: {deteccion['prioridad']}")
            else:
                logger.info("No se detecto contenido violento")
            
            resultado = {
                "es_violento": es_violento,
                "probabilidad_violencia": probabilidad_violencia,
                "detalles_violencia": violencia_detectada,
                "total_categorias_analizadas": len(candidate_labels),
                "categorias_encontradas": [v['label'] for v in violencia_detectada]
            }
            
            logger.info(f"RESULTADO VIOLENCIA: es_violento={es_violento}, prob={probabilidad_violencia:.4f}")
            return resultado
            
        except Exception as e:
            logger.error(f"ERROR ANALIZANDO VIOLENCIA: {e}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {
                "es_violento": False, 
                "probabilidad_violencia": 0.0, 
                "error": str(e)
            }

class ImageAnalyzer:
    def __init__(self):
        self.weapon_detector = WeaponDetector()
        self.violence_detector = ViolenceDetector()
        self.cargado = False

    def load_models(self):
        """Carga todos los modelos necesarios"""
        logger.info("INICIANDO CARGA DE MODELOS ESPECIALIZADOS")
        try:
            logger.info("Cargando detector de violencia...")
            self.violence_detector.load_model()
            
            logger.info("Cargando detector de armas...")
            self.weapon_detector.load_model()
            
            self.cargado = self.weapon_detector.cargado and self.violence_detector.cargado
            
            if self.cargado:
                logger.info("TODOS LOS MODELOS CARGADOS CORRECTAMENTE")
                logger.info(f"   - Detector de violencia: {self.violence_detector.model_name}")
                logger.info(f"   - Detector de armas: {self.weapon_detector.model_name}")
            else:
                logger.error("FALLO LA CARGA DE ALGUN MODELO")
                
        except Exception as e:
            logger.error(f"ERROR CARGANDO MODELOS: {e}")
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            self.cargado = False

    def analyze_image(self, image_path: str):
        """Analiza una imagen para contenido inapropiado"""
        if not self.cargado:
            return {"es_apto": False, "error": "Modelos no cargados", "puntuacion_riesgo": 1.0}

        try:
            logger.info(f"INICIANDO ANALISIS DE IMAGEN: {image_path}")
            
            if not os.path.exists(image_path):
                return {"es_apto": False, "error": "Archivo no encontrado", "puntuacion_riesgo": 1.0}

            logger.info("Ejecutando analisis de violencia...")
            resultado_violencia = self.violence_detector.analyze_violence(image_path)
            
            logger.info("Ejecutando analisis de armas...")
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

            # ✅ UMBRALES MÁS ESTRICTOS PARA RECHAZAR IMÁGENES CON ARMAS
            es_apto = not (
                # ✅ VIOLENCIA - UMBRAL BAJADO (0.4 en lugar de 0.7)
                (resultado_violencia.get("es_violento", False) and riesgo_violencia > 0.4) or  
                
                # ✅ ARMAS - UMBRAL MUY BAJO (0.2 en lugar de 0.6)
                (resultado_armas.get("armas_detectadas", False) and riesgo_armas > 0.2) or     
                
                # ✅ ARMAS EN ANÁLISIS DE VIOLENCIA - UMBRAL BAJO (0.15 en lugar de 0.5)
                (armas_en_violencia and confianza_armas_violencia > 0.15)                       
            )
            
            puntuacion_riesgo = max(riesgo_violencia, riesgo_armas)
            
            # Log del resultado
            logger.info("RESULTADOS OBTENIDOS:")
            logger.info(f"    Violencia: es_violento={resultado_violencia.get('es_violento')}, prob={riesgo_violencia:.4f}")
            logger.info(f"    Armas: detectadas={resultado_armas.get('armas_detectadas')}, confianza={riesgo_armas:.4f}")
            
            if not es_apto:
                razones = []
                if resultado_violencia.get("es_violento") and riesgo_violencia > 0.4:
                    razones.append(f"violencia ({riesgo_violencia:.4f})")
                if resultado_armas.get("armas_detectadas") and riesgo_armas > 0.2:
                    razones.append(f"armas YOLO/CLIP ({riesgo_armas:.4f})")
                if armas_en_violencia and confianza_armas_violencia > 0.15:
                    razones.append(f"armas en violencia ({confianza_armas_violencia:.4f})")
                
                logger.warning(f"IMAGEN RECHAZADA - Puntuacion riesgo: {puntuacion_riesgo:.4f}")
                logger.warning(f"    - Razon: {'; '.join(razones)}")
            else:
                logger.info(f"IMAGEN APROBADA - Puntuacion riesgo: {puntuacion_riesgo:.4f}")
            
            resultado_final = {
                "es_apto": es_apto,
                "analisis_violencia": resultado_violencia,
                "analisis_armas": resultado_armas,
                "puntuacion_riesgo": float(puntuacion_riesgo),
                "armas_detectadas_en_violencia": armas_en_violencia,
                "confianza_armas_violencia": float(confianza_armas_violencia)
            }
            
            logger.info(f"RESUMEN FINAL: es_apto={es_apto}, riesgo={puntuacion_riesgo:.4f}")
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