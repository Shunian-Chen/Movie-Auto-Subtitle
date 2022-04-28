import os
import torch
import argparse
import pandas as pd
import soundfile as sf
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
BASE = os.path.abspath(os.path.join(os.path.abspath("."), ".."))
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
MODEL_PATH = os.path.join(BASE, "wav2vec2-base-960h-finetune", "checkpoint-19600")
# load pretrained model
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
PROCESSOR = Wav2Vec2Processor.from_pretrained(MODEL_ID)
MODEL = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(DEVICE)

def parse_time(t):
    start_time, end_time = map(int, t.split("_"))
    start_time /= 1000
    start_hour = start_time//3600%24
    start_min = start_time//60%60
    start_sec = start_time%60
    start = "%02d:%02d:%06.3f"%(start_hour, start_min, start_sec)
    
    end_time /= 1000
    end_hour = end_time//3600%24
    end_min = end_time//60%60
    end_sec = end_time%60
    end = "%02d:%02d:%06.3f"%(end_hour, end_min, end_sec)

    return f"{start} --> {end}".replace(".", ",")
    
def trans2flac(audio_path, audio_name, output_path, format):
    audio_type = audio_name.split(".")[-1][1:]
    exp_file = audio_name.split(".")[0]
    audio = AudioSegment.from_file(os.path.join(audio_path, audio_name), format=audio_type)
    audio.export(f'{os.path.join(output_path, exp_file)}.{format}', format = str(format))
    return f"{exp_file}.{format}"

def speech2text(path):
    # load audio
    audio_input, sample_rate = sf.read(path)
    # pad input values and return pt tensor
    input_values = PROCESSOR(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values.to(DEVICE)
    # retrieve logits & take argmax
    pred = MODEL(input_values)
    logits = pred.logits
    predicted_ids = torch.argmax(logits, dim=-1)[0]

    # transcribe
    transcription = PROCESSOR.decode(predicted_ids)

    return transcription

def speech_recoginze(audio_name, base):
    AUDIO_FOLDER = os.path.join(base, "Data", "outputs", audio_name, "Preprocess", "Split")
    CLEAN_SPEECH_PATH = os.path.join(base, "Data", "Clean splits", audio_name)
    TRANSCIPTION_PATH = os.path.join(base, "Data", "Transciption")

    if not os.path.exists(CLEAN_SPEECH_PATH):
        os.mkdir(CLEAN_SPEECH_PATH)
    

    #Load audio names
    AUDIO_NAMES = []
    for _, _, files in os.walk(AUDIO_FOLDER):
        AUDIO_NAMES += files
    
    #record the time duration of audio
    AUDIO_TIME = list([parse_time(time.split(".")[0]) for time in AUDIO_NAMES])
    
    #transfer wav file to flac file
    CLEAN_SPEECH = [trans2flac(AUDIO_FOLDER, audio_name, CLEAN_SPEECH_PATH, "flac") for audio_name in AUDIO_NAMES]
    
    #concadenate foler path and audio name
    AUDIO_PATH = [os.path.join(CLEAN_SPEECH_PATH, audio_name) for audio_name in CLEAN_SPEECH]
    
    #save information
    subtitles = pd.DataFrame(columns=["Time" , "Path", "Text"])
    subtitles["Time"] = AUDIO_TIME
    subtitles["Path"] = AUDIO_PATH

    #speech to text
    subtitles["Text"] = subtitles["Path"].apply(lambda x: speech2text(x))

    #save results
    filename = f"{audio_name}.srt"
    f = open(os.path.join(TRANSCIPTION_PATH, filename), "w")
    for idx, row in subtitles.iterrows():
        f.write(f'{idx+1}\n{row["Time"]}\n{row["Text"]}\n\n')
    f.close()


if __name__ == "__main__":
    #get audio name
    base = os.path.abspath(os.path.join(os.path.abspath("."), ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
                        type=str)
    arg = parser.parse_args()
    speech_recoginze(arg.filename, base)

    
    

