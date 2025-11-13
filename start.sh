#!/bin/bash
echo "🚀 Iniciando servidor de modelos en Render..."
echo "📁 Directorio: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "🔧 Instalando dependencias..."

# Instalar dependencias
pip install -r requirements.txt

# Crear directorio para logs si no existe
mkdir -p logs

echo "✅ Dependencias instaladas"
echo "🎯 Iniciando servidor..."
python modelo_server.py