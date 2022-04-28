import os
import gc
import pydub
import argparse
import subprocess
from pydub import AudioSegment

#global variables
SAMPLE_RATE = 16000

def extract_audio(videos_file_path, file_name, export_path, format = "wav"):
    try:
        audio_name = file_name.split(".")[0]
        video_path = os.path.join(videos_file_path, file_name)
        audio_path = os.path.join(export_path, audio_name) + f".{format}"
        cmd = f'ffmpeg -i \"{video_path}\" -f \"{format}\" -vn -ac 2 -y \"{audio_path}\"'
        subprocess.call(cmd)
    except Exception as ex:
        print("Error: ", ex)
    
    return audio_name + f".{format}"

def normalize(audio_path, audio_file, output_path, format = "wav"):
    try:
        audio_name = audio_file.split(".")[0]
        input_path = os.path.join(audio_path, audio_file)
        output_path = f"{os.path.join(output_path, audio_name)}-normalized.{format}"
        cmd = f'ffmpeg-normalize \"{input_path}\" -o \"{output_path}\" --sample-rate {SAMPLE_RATE}'
        subprocess.call(cmd)
    except Exception as ex:
        print("Error: ", ex)

def vocal_spearation(audio_path, output_path, base):
    try:
        inference = os.path.join(base, "Code", "inference.py")
        model = os.path.join(base, "Code", "models", "baseline.pth")
        cmd = f'python \"{inference}\" --input \"{audio_path}\" --output \"{output_path}\" --pretrained_model \"{model}\" --gpu 0 -B 4 --tta'
        subprocess.call(cmd)
    except Exception as ex:
        print("Error: ", ex)

def split_audio(audio_path, audio_name, min_silence_len = 1000, silence_thresh = -16, keep_silence = 500):
    #read audio
    audio_type = os.path.splitext(audio_name)[-1][1:]
    audio = AudioSegment.from_file(os.path.join(audio_path, audio_name), format = audio_type)

    #normalize audio
    # normalized_audio = match_target_amplitude(audio, silence_thresh)

    #create folder to store the result segments if not exist
    folder = os.path.join(audio_path, "Split")
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    #split segment
    not_silence_ranges = pydub.silence.detect_nonsilent(audio, min_silence_len = min_silence_len, silence_thresh = silence_thresh, seek_step = 1)

    #cut the slice from the original audio and save it
    for idx in range(len(not_silence_ranges)):
        current_start_pos = max(0, not_silence_ranges[idx][0] - keep_silence)
        # current_end_pos = round(not_silence_ranges[idx][1])
        current_end_pos = round(not_silence_ranges[idx][1]) if idx == len(not_silence_ranges)-1 else not_silence_ranges[idx + 1][0]-keep_silence

        new = audio[current_start_pos:current_end_pos] 

        #segment is too small
        if len(new) <= 500:
            continue
        
        #name the segment using its time
        file_name = f"{current_start_pos}_{current_end_pos}.{audio_type}"
        save_name = os.path.join(folder, file_name)
        new.export(save_name, format = audio_type)
    audio = audio.empty()

#util functions

def time_trans(t):
    h = t//3600
    m = t//60
    s = int(t%60)
    ms = round(t - int(t), 3) * 1000
    return "%02d-%02d-%02d,%03d" % (h, m, s, ms)

#adjust target amplitude
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def extract(video_path, base):
    video_name = video_path.split(".")[0]
    audio_format = "wav"

    #set paths
    DATA_PATH = os.path.join(base, 'Data', 'videos', 'original')
    AUDIO_PATH = os.path.join(base, 'Data', 'outputs', video_name)
    MIN_SILENCE_LEN = 1000
    SILENCE_THRESH = -30
    
    if not os.path.exists(AUDIO_PATH):
        os.mkdir(AUDIO_PATH)
    
    output_path = os.path.join(AUDIO_PATH, "Preprocess")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    audio_name = video_name
    #extract audio
    print("Extracting audio..")
    audio_file = extract_audio(DATA_PATH, video_path, output_path, audio_format)
    print("Done")
    
    #separate vocal from audio
    print("Separating...")
    audio_path = os.path.join(output_path, audio_file)
    vocal_spearation(audio_path, output_path, base)
    print("Done")

    #normalize separated audio
    print("Normalizing...")
    vocal_file = f"{audio_name}_Vocals.{audio_format}"
    normalize(output_path, vocal_file, output_path, audio_format)
    normal_vocal_name = f"{audio_name}_Vocals-normalized.{audio_format}"
    print("Done")

    

    #split audio into pieces
    print("Spliting")
    split_audio(output_path, normal_vocal_name, MIN_SILENCE_LEN, SILENCE_THRESH)
    print("Done")

if __name__ == "__main__":
    #get video name
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
                        type=str)
    arg = parser.parse_args()

    extract(arg.filename)