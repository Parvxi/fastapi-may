import signal
import threading
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel
from typing import Annotated
import uvicorn
from OSC_Receiver_Simple import EEGProcessor
import models as models
from database import engine, SessionLocal
from sqlalchemy.orm import Session
from typing import List
from pythonosc import dispatcher, osc_server
from FE import FE  # Import your FE class
from prediction import predic  # Import your predic class
from threading import Thread


app = FastAPI()

# Create the database models and tables
models.Base.metadata.create_all(bind=engine)

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Depends(get_db)

# Define the request models
class HistoryItem(BaseModel):
    timestamp: str
    duration: str
    result: str

# Initialize your featureObj and predic objects
featureObj = FE()
predic_obj = predic()
eeg_processor = EEGProcessor(featureObj, predic_obj)

# Define the FastAPI endpoints
@app.get("/history", response_model=List[HistoryItem])
async def get_history(db: Session = db_dependency):
    history_items = db.query(models.History).all()
    return [{"timestamp": item.timestamp, "duration": item.duration, "result": item.result} for item in history_items]

@app.get("/history/{item_id}", response_model=HistoryItem)
async def get_history_item(item_id: int, db: Session = db_dependency):
    history_item = db.query(models.History).filter(models.History.id == item_id).first()
    if not history_item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History item not found")
    return {"timestamp": history_item.timestamp, "duration": history_item.duration, "result": history_item.result}

@app.get("/history/{item_id}/result", response_model=str)
async def get_history_result(item_id: int, db: Session = db_dependency):
    history_item = db.query(models.History).filter(models.History.id == item_id).first()
    if not history_item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History item not found")
    return history_item.result

import threading
import time
def run_eeg_processor(dispatcher):
    ip = "0.0.0.0"
    port = 5000

    # Initialize your featureObj and predic objects
    featureObj = FE()
    predic_obj = predic()

    eeg_processor = EEGProcessor(featureObj, predic_obj)

    dispatcher.map("/muse/eeg", eeg_processor.on_new_eeg_data)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port " + str(port))

    def stop_server():
        """
        Function to stop the server
        """
        print("Stopping server...")
        server.shutdown()
        server.server_close()

    # Set a timer to stop the server after 30 seconds
    timer = threading.Timer(30, stop_server)
    timer.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        timer.cancel()  # Cancel the timer
        # Insert last prediction into the database
        if eeg_processor.last_prediction is not None:
            eeg_processor.insert_prediction_to_db(eeg_processor.last_prediction)

    print("Script finished running")


@app.post("/start_eeg_processing")
async def trigger_eeg_processor():
    dispatcher_obj = dispatcher.Dispatcher()
    threading.Thread(target=run_eeg_processor, args=(dispatcher_obj,)).start()
    return JSONResponse(content={"message": "EEG processor started"})


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
