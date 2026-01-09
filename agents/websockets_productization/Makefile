.PHONY: help build up down logs test clean

help:
	@echo "Available commands:"
	@echo "  make build    - Build the Docker images"
	@echo "  make up       - Start the services"
	@echo "  make down     - Stop the services"
	@echo "  make logs     - View logs"
	@echo "  make test     - Send a test request to the orchestrator"
	@echo "  make clean    - Remove containers and volumes"

build:
	docker compose build

up:
	docker compose up

down:
	docker compose down

logs:
	docker compose logs -f

test:
	@echo "Testing Triage Endpoint..."
	curl -X POST http://localhost:8000/triage \
		-H "Content-Type: application/json" \
		-d '{"symptoms": "I have a mild headache and a runny nose."}'

clean:
	docker compose down -v --remove-orphans
