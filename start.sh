#!/bin/bash
# Install SpaCy model
python -m spacy download en_core_web_sm

# Start the FastAPI app with gunicorn
<<<<<<< HEAD
gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT
=======
gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT
>>>>>>> 1c086e086f798b5ca4c32b1451b08f0a6f05a16c
