import os
import argparse
import ffmpeg

def add_subtitle(file, base):
    SRC_VIDEO = os.path.join(base, "Data", "videos", "original", f"{file}.mp4")
    SRC_SCRIPT = os.path.join(base, "Data", "Transciption", f"{file}.srt")
    SAVE_FOLDER = os.path.join(base, "Data", "videos", "subtitled", file)
    
    video = ffmpeg.input(SRC_VIDEO)
    audio = video.audio
    ffmpeg.concat(video.filter("subtitles", "Interstellar.srt"), audio, v = 1, a = 1).output(f"{SAVE_FOLDER}_subtitled.mp4").run()

if __name__ == "__main__":
    #get video name
    base = os.path.abspath(os.path.join(os.path.abspath("."), ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
                        type=str)
    arg = parser.parse_args()

    add_subtitle(arg.filename, base)