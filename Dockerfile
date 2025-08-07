FROM python:3.11-slim

# Diretório de trabalho
WORKDIR /app

# Copia arquivos necessários
COPY . /app

# Instala dependências
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expõe a porta que o uvicorn vai usar
EXPOSE 8080

# Comando para rodar a API (Cloud Run espera porta 8080)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
