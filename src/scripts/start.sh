#!/bin/bash
echo "🚀 Iniciando servidor de modelos en Render..."
echo "📁 Directorio: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "🔧 Instalando dependencias optimizadas..."

# Crear directorios necesarios
mkdir -p logs
mkdir -p uploads
mkdir -p /tmp/model_cache

# Instalar dependencias
pip install -r requirements.txt

# Verificar que los modelos pueden cargarse
echo "🔍 Verificando importaciones..."
python -c "
try:
    from analisis_imagen import ImageAnalyzer
    print('✅ ImageAnalyzer importado correctamente')
except Exception as e:
    print(f'❌ Error importando ImageAnalyzer: {e}')
"

echo "✅ Dependencias instaladas"
echo "🎯 Iniciando servidor optimizado..."
echo "📊 Health check disponible en: http://localhost:10000/health"

# Iniciar servidor
exec python modelo_server.py