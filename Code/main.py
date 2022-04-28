import argparse
from Code.extract_audio import extract
from Code.speech_recognize import speech_recoginze
from Code.add_subtitle import add_subtitle

def auto_subtitle(filename, base):
    extract(filename, base)
    speech_recoginze(filename.split(".")[0], base)
    add_subtitle(filename.split(".")[0], base)

if __name__ == "__main__":
    #get video name
    base = os.path.abspath(os.path.join(os.path.abspath("."), ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
                        type=str)
    arg = parser.parse_args()

    filename = arg.filename
    auto_subtitle(filename, base)