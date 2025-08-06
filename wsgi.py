from asgi2wsgi import ASGI2WSGI
from main import app  # importa seu FastAPI app

# Adaptador ASGI para WSGI (usado pelo PythonAnywhere)
application = ASGI2WSGI(app)
