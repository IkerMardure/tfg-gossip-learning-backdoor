# Nueva Estructura del Proyecto

## Resumen
El proyecto ha sido reorganizado en 4 carpetas principales:

```
GLow_TFG/
├── code/                    # Código fuente (GLow-master)
│   └── GLow-master/
│       ├── client.py
│       ├── client_backdoor.py
│       ├── main_backdoor.py
│       ├── conf/
│       ├── visualization/
│       └── ... (resto del código)
│
├── data/                    # Datos y datasets
│   └── datasets/
│       └── MNIST/
│
├── results/                 # Resultados de simulaciones
│   └── outputs/
│       └── 2026-04-28 - 17_12 - Ring_5/
│       └── ... (resto de outputs)
│
├── paper/                   # Documentación (LaTeX)
│   ├── main.tex
│   ├── main.pdf
│   ├── chapters/
│   ├── figures/
│   └── ...
│
└── [archivos de config]
    ├── .git/
    ├── .venv/
    └── ...
```

## Cambios de Paths a Realizar

### 1. dataset.py
**Antiguo:**
```python
DATASET_DIR = Path(__file__).parent.parent / "datasets" / dataset_name
```

**Nuevo:**
```python
DATASET_DIR = Path(__file__).parent.parent.parent / "data" / "datasets" / dataset_name
```

### 2. Paths en main_backdoor.py
Si hay referencias hardcodeadas a outputs:
**Antiguo:**
```python
output_path = Path(__file__).parent / "outputs"
```

**Nuevo:**
```python
output_path = Path(__file__).parent.parent.parent / "results" / "outputs"
```

### 3. Paths en visualization/
Si hay scripts que leen de outputs:
**Antiguo:**
```python
output_dir = Path(__file__).parent.parent / "outputs"
```

**Nuevo:**
```python
output_dir = Path(__file__).parent.parent.parent.parent / "results" / "outputs"
```

## Cómo Ejecutar con la Nueva Estructura

### Cambiar a la carpeta de código
```bash
cd code/GLow-master
```

### Ejecutar simulación
```bash
python main_backdoor.py conf/base.yaml mnist_test .\conf\topologies\analysis\Ring\ring_5.yaml
```

### Compilar el paper
```bash
cd ../../paper
latexmk -pdf -shell-escape main.tex
```

### Generar gráficos
```bash
cd ../../code/GLow-master
python .\visualization\plot_accuracies_per_node.py "..\..\results\outputs\2026-04-28 - 17_12 - Ring_5\mnist_test_raw.out"
```

## Nota Importante
Los scripts de Python deben actualizarse para usar las nuevas rutas. Usa `Path(__file__).parent` junto con `.parent` múltiples veces para navegar correctamente desde el código hasta las carpetas de datos y resultados.
