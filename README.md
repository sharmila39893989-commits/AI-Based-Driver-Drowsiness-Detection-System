# JAAGRATA driver safety website

The React frontend provides Dashboard, Live Detection, Analytics, Alerts,
History, and About pages. The FastAPI backend handles image analysis.

## Run locally in development

Start the backend from the project root:

```powershell
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8080
```

In a second PowerShell window, install the frontend packages and start Vite:

```powershell
cd frontend
npm install
npm run dev
```

Vite prints its URL in the terminal (normally
[http://127.0.0.1:5173/](http://127.0.0.1:5173/)) and does not open a browser
automatically. Open the printed URL when you are ready. API requests are
proxied to the backend on port 8080.

## Serve the frontend through FastAPI

Build the frontend:

```powershell
cd frontend
npm run build
```

Then start FastAPI from the project root using the command above and open
[http://127.0.0.1:8080/](http://127.0.0.1:8080/). The built React pages are
served at the root and their client-side routes also work when refreshed.
