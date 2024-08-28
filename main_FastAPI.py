from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from fastapi import FastAPI, File, UploadFile
from typing import List

# Step 2: Load the model and processor
local_path = r"E:\GIT\LUXEMBOURGISH-TTS\Model"
processor2 = AutoProcessor.from_pretrained(local_path)
model2 = AutoModelForSpeechSeq2Seq.from_pretrained(local_path)

print("Model and processor loaded successfully from local path")

# Step 3: FastAPI Integration
app = FastAPI()

@app.post("/whisper")
async def transcribe(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        audio = await file.read()
        inputs = processor2(audio, return_tensors="pt", sampling_rate=16000)
        generated_ids = model2.generate(inputs["input_features"])
        transcription = processor2.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.append({"filename": file.filename, "transcript": transcription})
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
