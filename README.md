# Luxembourgish Speech-to-Text

**Speech AI · Wav2Vec2 fine-tuning · Whisper inference · GPU/HPC performance analysis**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ibPONCmdKUyjRJVcOE8fUN61mSMEbM2n?usp=sharing)

**[Open the original Google Colab notebook](https://colab.research.google.com/drive/1ibPONCmdKUyjRJVcOE8fUN61mSMEbM2n?usp=sharing)** · [Open the repository notebook in Colab](https://colab.research.google.com/github/parsavares/Luxembourgish-STT/blob/main/research.ipynb) · [View notebook on GitHub](research.ipynb)

A research project exploring **automatic speech recognition (ASR) for Luxembourgish**, from audio preprocessing and model evaluation to GPU inference, a Wav2Vec2 fine-tuning workflow, and web integration prototypes.

The core work lives in **Google Colab** and the accompanying [research notebook](research.ipynb). It compares Luxembourgish-adapted Wav2Vec2 and Whisper models, measures word error rate and CPU/GPU latency, and investigates the practical memory constraints of adapting a large speech model.

[Start in Colab](#start-in-google-colab) · [Explore the experiments](#research-workflow) · [Recorded results](#recorded-results) · [Fine-tuning](#wav2vec2-fine-tuning) · [Repository guide](#repository-guide)

## Start in Google Colab

**Colab is the primary research environment.** Start with the notebook to explore the models and experiments; the FastAPI and Gradio files are separate integration prototypes.

1. **Open a notebook.** Use the original Colab link above, or open the repository copy directly in Colab. Create your own working copy.
2. **Select a GPU runtime.** Open **Runtime → Change runtime type** and choose a GPU accelerator. The saved notebook metadata specifies a T4; available hardware depends on your Colab session. See the [Colab resource FAQ](https://research.google.com/colaboratory/faq.html#resource-limits).
3. **Prepare the environment.** Run the installation cells needed for the experiment you choose. The notebook uses PyTorch, Transformers, Torchaudio, Librosa, Datasets, JiWER, Pandas, and Matplotlib. Package versions are not pinned, so historical API calls may need adjustment.
4. **Configure authentication and data paths.** Replace inline credentials with your own authentication, and update the notebook's Google Drive paths for experiments using local audio and transcripts.
5. **Run the relevant sections in order.** Sections A/B introduce inference, C compares the models, and D contains the fine-tuning workflow. Review each section before executing it; later sections depend on earlier variables and external data.

Check GPU access in a notebook cell:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

For resources requiring Hugging Face authentication, remove the notebook's inline token assignments and use the [interactive login](https://huggingface.co/docs/huggingface_hub/package_reference/authentication):

```python
from huggingface_hub import login

login()
```

Use your own access credentials and obtain any required dataset permissions. The historical notebook contains hard-coded credentials; they should be removed, and any still-active exposed token should be revoked.

### Choose your starting point

| Goal | Notebook section | What you need |
| --- | --- | --- |
| Transcribe a microphone recording | **A / B** | Browser microphone access and the selected pretrained model |
| Compare transcription quality | **C** | Paired audio files and reference transcripts |
| Explore GPU inference | **C → GPU for whisper** | A CUDA-enabled runtime and the evaluation audio |
| Inspect or adapt fine-tuning | **D → Wav2Vec2** | Training data, compatible dependencies, and sufficient GPU memory |
| Explore chatbot integration | **F** | Speech model plus updated audio/API integration code |

## Research workflow

| Area | Work included |
| --- | --- |
| **Audio preparation** | Browser recording, mono conversion, resampling to 16 kHz, waveform visualization, and transcript loading |
| **Model comparison** | Wav2Vec2 CTC decoding and Whisper sequence-to-sequence transcription |
| **Evaluation** | Raw and normalized word error rate (WER), prediction inspection, and export of comparison data |
| **GPU computing** | Explicit CUDA device placement and CPU/GPU inference timing comparisons |
| **Fine-tuning** | Wav2Vec2 dataset preparation, a custom CTC padding collator, and Hugging Face Trainer configuration with FP16 |
| **Application integration** | FastAPI transcription endpoint, Gradio microphone interface, and a speech-to-chatbot experiment |

### Models and data

The project builds on existing Luxembourgish speech models. Their original training and checkpoints belong to the respective model authors.

| Component | Resource | Role in this repository |
| --- | --- | --- |
| **Wav2Vec2 XLSR** | [Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-14h](https://huggingface.co/Lemswasabi/wav2vec2-large-xlsr-53-842h-luxembourgish-14h) | Baseline ASR evaluation and further fine-tuning workflow |
| **Whisper Large** | [ZLSCompLing/whisper_large_lb_ZLS_v4_38h](https://huggingface.co/ZLSCompLing/whisper_large_lb_ZLS_v4_38h) | Baseline evaluation, GPU inference, and integration experiments |
| **Training dataset** | [ZLSCompLing/validator_38h](https://huggingface.co/datasets/ZLSCompLing/validator_38h) | Dataset used by the Wav2Vec2 fine-tuning setup |
| **Evaluation audio** | Local `kapitel_1` recordings and transcripts referenced in the notebook | Model-quality and inference-latency comparisons |

The local evaluation recordings are not bundled with this repository. To reproduce that workflow with your own data, provide audio files and a **headerless, tab-separated transcript file** with two columns: filename and reference transcription. Update both `transcript_path` and the audio directory inside the prediction functions.

## Recorded results

Results below come from saved outputs and timing arrays in [research.ipynb](research.ipynb). They describe the original experiments and have not been rerun for this documentation.

### Transcription quality

Lower WER is better. These are **baseline comparisons of the existing pretrained models**, before this project's further fine-tuning attempt.

| Model | Raw WER | Normalized WER |
| --- | ---: | ---: |
| Wav2Vec2 XLSR | 0.68 | 0.28 |
| Whisper Large | **0.33** | **0.18** |

Normalization lowercases text, removes punctuation, collapses repeated spaces, and trims whitespace. Using the rounded notebook values, Whisper's normalized WER is **10 percentage points lower**, approximately a **35.7% relative reduction** compared with Wav2Vec2 on this evaluation set.

These measurements depend on the recordings, reference transcripts, and normalization procedure. They do not establish performance across all Luxembourgish speakers or domains.

### CPU versus GPU inference

The notebook stores **100 latency measurements for each execution mode** of the Whisper experiment:

| Execution mode | Mean recorded time per audio file |
| --- | ---: |
| CPU | 204.524 s |
| GPU | **3.607 s** |

The ratio of these means is approximately **56.7×**. This is an observed timing ratio from the saved experiment; hardware details and execution conditions are not fully documented, and the CPU/GPU timing blocks have slightly different boundaries. It is not a claim of multi-GPU or cluster scaling.

## Wav2Vec2 fine-tuning

Section D configures further adaptation of the Luxembourgish Wav2Vec2 model using `ZLSCompLing/validator_38h`.

The workflow prepares audio inputs and transcription labels, dynamically pads batches, masks label padding with `-100`, and invokes Hugging Face `Trainer`.

| Setting | Notebook configuration |
| --- | --- |
| Training / evaluation batch size | 8 per device |
| Training epochs | 3 |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Precision | FP16 enabled |
| Evaluation strategy | Every epoch |
| Checkpoint retention | 2 |

**Recorded status:** the saved training attempt ends with a CUDA out-of-memory error. The notebook does not include completed post-fine-tuning evaluation results. Whisper is used for inference and evaluation; a completed Whisper fine-tuning experiment is not included.

For another training attempt, begin with a smaller per-device batch size and assess memory use before scaling up. Gradient accumulation and gradient checkpointing are possible extensions to evaluate; they are not enabled in the committed training configuration.

## GPU computing and HPC relevance

This project connects model development with practical compute decisions: moving models and inputs onto CUDA devices, measuring inference latency, configuring mixed-precision training, and diagnosing GPU memory limits.

The recorded environment is **Google Colab with CUDA**. The repository provides a starting point for porting the experiments to an HPC GPU environment; Slurm launchers, distributed training, and multi-node benchmarks are not included.

## Repository guide

| File or directory | Purpose |
| --- | --- |
| [research.ipynb](research.ipynb) | Main research notebook: inference, evaluation, CPU/GPU comparison, fine-tuning setup, and chatbot exploration |
| [main_FastAPI.py](main_FastAPI.py) | Prototype `POST /whisper` endpoint accepting uploaded audio files |
| [main_Gradio.py](main_Gradio.py) | Prototype browser microphone interface for transcription |
| [Install-libraries.py](Install-libraries.py) | Basic dependency installation helper for the integration scripts |
| [Model/](Model/) | Whisper configuration, tokenizer, preprocessing metadata, and a weight index; the referenced model-weight shards are not included |

### Web integration status

The application scripts need adaptation before use. Replace the machine-specific model path with a downloaded model directory or the appropriate Hugging Face model ID. Decode uploaded audio bytes or file paths into a waveform and resample to 16 kHz before passing it to the processor. The Gradio script also uses a legacy audio-component API.

The installer lists the integration libraries but does not cover all notebook dependencies. FastAPI file uploads additionally require `python-multipart`. These scripts are experimental entry points; the notebook is the primary source for the research workflow.

## Reproducibility and next steps

To turn the recorded experiments into a repeatable benchmark, the main next steps are:

- Pin a compatible Python/package environment and document GPU, CPU, and driver details.
- Supply a shareable evaluation manifest with dataset provenance and permitted access.
- Align timing boundaries, add warm-up runs, and synchronize GPU work around measurements.
- Complete a memory-aware fine-tuning run and evaluate it on a separate held-out set.
- Update the web interfaces and add cluster launch scripts if porting to HPC infrastructure.

## Author and acknowledgements

**[Parsa Vares](https://github.com/parsavares)**

This repository documents experiments and integrations using the pretrained models and dataset linked above. Credit for those resources belongs to **Lemswasabi** and **ZLSCompLing**. Consult their model and dataset pages for applicable terms and access requirements.
