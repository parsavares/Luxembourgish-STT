from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import gradio as gr

# Step 2: Load the model and processor
local_path = r"E:\GIT\LUXEMBOURGISH-TTS\Model"
processor2 = AutoProcessor.from_pretrained(local_path)
model2 = AutoModelForSpeechSeq2Seq.from_pretrained(local_path)

print("Model and processor loaded successfully from local path")

# Step 3: Gradio Integration
def transcribe_audio(audio):
    inputs = processor2(audio, return_tensors="pt", sampling_rate=16000)
    generated_ids = model2.generate(inputs["input_features"])
    transcription = processor2.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Define the Gradio interface
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.inputs.Audio(source="microphone", type="filepath"),
    outputs="text",
    live=True
)

# Launch the Gradio app
iface.launch()
