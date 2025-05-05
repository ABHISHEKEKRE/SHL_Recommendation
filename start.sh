#!/bin/bash
# Install SpaCy model
python -m spacy download en_core_web_sm

# Start the FastAPI app with gunicorn
gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT
=======
gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT
