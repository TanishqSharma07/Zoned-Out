# import whisper
from faster_whisper import WhisperModel


# model = whisper.load_model("tiny")
model = WhisperModel("base", compute_type="int8")


def transcribe_audio(file_path):
    # result = model.transcribe(file_path)
    # return result["text"]

    segments, _ = model.transcribe(file_path)
    return " ".join([segment.text for segment in segments])
