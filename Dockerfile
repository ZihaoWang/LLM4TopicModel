FROM python:3.10

COPY requirements.txt app/requirements.txt

# The /app directory should act as the main application directory
WORKDIR /app

RUN pip install -r requirements.txt

COPY . .

CMD ["bash"]
