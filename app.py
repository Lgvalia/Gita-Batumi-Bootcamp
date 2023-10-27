from fastapi import (FastAPI, 
                     File, 
                     UploadFile)
import pandas as pd
import pickle
from fastapi.responses import FileResponse
import tempfile


app = FastAPI()


@app.post('/predict')
def predict_csv(input_file: UploadFile = File(...)):
    """post request for predicted from csv file into output"""    

    df = pd.read_csv(input_file.file)    

    # load model
    pkl_model = pickle.load(open('pipeline.pkl', 'rb'))

    #predict
    predictions = pkl_model.predict(df).astype(int).tolist()
    df['prediction'] = predictions
    
    # Create a temporary file to save the results
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        # Save the DataFrame to the temporary file
        df.to_csv(temp_file.name, index=False)

    # Return the temporary file as a response
    return FileResponse(temp_file.name, filename="predictions.csv", media_type="application/octet-stream")