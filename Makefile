install:
	# Install dependencies in the current environment
	pip install -r requirements.txt

run:
	python naradio.py --mode webcam

run-web:
	cp -r web /app || true
	gunicorn -b 0.0.0.0:5000 web.naradio_web:app

docker-up:
	docker compose up --build

docker-up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.gpu.yml up --build

gpu-info:
	python naradio.py --gpu-info

pip-venv-setup:
	./setup_pip_venv.sh
