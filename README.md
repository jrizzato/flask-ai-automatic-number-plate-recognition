## Activación del entorno virtual
```bash
.venv\Scripts\Activate.ps1
```


## Instalar Label Studio

```bash
# Requires Python >=3.8
pip install label-studio

# Start the server at http://localhost:8080
label-studio
```
pip freeze > deps/requirements.txt
pip install -r deps/requirements.txt

tensorboard --logdir="object_detection" 

python -m pip install --upgrade pip setuptools wheel                                                        