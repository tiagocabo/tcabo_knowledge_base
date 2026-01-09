# WebSocket Productization Architecture

This project demonstrates a microservices architecture using **FastAPI**, **LangGraph**, and **WebSockets** to simulate a medical triage system. The system consists of an orchestrator service that classifies patient symptoms and routes them to the appropriate specialist (Doctor Service) via WebSockets.

## Architecture Overview

The system is composed of three main services running in Docker containers:

1.  **Orchestrator Service** (`orchestrator`):
    -   **Port**: 8000
    -   **Role**: Acts as the main entry point (API Gateway).
    -   **Workflow**:
        1.  Receives a POST request at `/triage` with patient symptoms.
        2.  Uses an LLM (LangGraph) to classify specific symptoms as "Urgent" or "Regular".
        3.  Establish a WebSocket connection to the appropriate Doctor Service based on classification.
        4.  Sends the symptoms and receives a structured treatment plan.
        5.  Returns the plan to the user.

2.  **Urgent Doctor Service** (`doctor-urgent`):
    -   **Port**: 8765 (Internal: 8000)
    -   **Type**: Urgent Care
    -   **Role**: specialized agent for urgent/emergency cases.
    -   **Communication**: WebSocket (`/ws`).

3.  **Regular Doctor Service** (`doctor-regular`):
    -   **Port**: 8766 (Internal: 8000)
    -   **Type**: Primary Care
    -   **Role**: specialized agent for general/non-urgent cases.
    -   **Communication**: WebSocket (`/ws`).

## Prerequisites

-   [Docker](https://www.docker.com/) installed.
-   `OPENAI_API_KEY` environment variable set in a `.env` file or your shell.

## specific Project Structure

```
agents/websockets_productization/
├── README.md               # This file
├── docker-compose.yml      # Service orchestration
├── Makefile                # Command shortcuts
├── doctor_service/         # Doctor Service code (Shared image)
│   ├── main.py
│   ├── models.py
│   └── Dockerfile
└── orchestrator_service/   # Orchestrator Service code
    ├── main.py
    ├── graph.py
    ├── models.py
    └── Dockerfile
```

## Setup & Running

1.  **Environment Setup**:
    Create a `.env` file in this directory with your OpenAI API Key:
    ```bash
    OPENAI_API_KEY=sk-...
    ```

2.  **Build the Services**:
    ```bash
    make build
    ```

3.  **Start the Services**:
    ```bash
    make up
    ```

4.  **View Logs** (Optional):
    ```bash
    make logs
    ```

5.  **Stop Services**:
    ```bash
    make down
    ```

## Testing

You can test the system using the provided Makefile command which sends a sample request to the orchestrator:

```bash
make test
```

Or manually using `curl`:

```bash
curl -X POST http://localhost:8000/triage \
    -H "Content-Type: application/json" \
    -d '{"symptoms": "I have a mild headache and a runny nose."}'
```

### Expected Output

The system will return a JSON object containing the Doctor's response and a detailed treatment plan structure:

```json
{
  "response": "Regular doctor plan received.",
  "plan": {
    "urgency": "regular",
    "summary": "...",
    "likely_causes": [...],
    "home_care": [...],
    "otc_options": [...],
    "what_to_avoid": [...],
    "red_flags": [...],
    "what_to_tell_clinician": [...],
    "disclaimer": "..."
  }
}
```
