FROM python:3.11-slim
#base image
WORKDIR /app
#container working directory
COPY requirements.txt .
#Copy project files
RUN pip install --no-cache-dir -r requirements.txt
#Execute command
COPY . .

EXPOSE 8000
#Port
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
#Container start up command