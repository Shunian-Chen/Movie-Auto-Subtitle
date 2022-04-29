# Movie-Auto-Subtitle

This project aims to provide a solution for applying deep-learning methods to add subtitles to movies. Given a movie without subtitles as input, the application will go through audio extraction, vocal separation, audio split on silence, speech recognition, and finally integrate the subtitle to the movie and output the movie with subtitle.



## Data Pipeline

![image-20220428201455883](C:\Users\14183\AppData\Roaming\Typora\typora-user-images\image-20220428201455883.png)



## Model

This project uses a pre-trained XLS-R model from Facebook, which makes it possible to be applied to multiple languages apart from English in the future. The pre-trained model can be found in [Hugging Face](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english). The model is trained on [Libspeech](https://www.openslr.org/12) and fine-tuned on [Common Voice](https://commonvoice.mozilla.org/en?gclid=CjwKCAjw9qiTBhBbEiwAp-GE0c2e2N4od2TDJXqZ6BF44n9I7ajd9MlAoV6L7aJd_dQCU3eUVgzOkxoCTKcQAvD_BwE). The paper for the model can be found [here](https://arxiv.org/abs/2111.09296). 

Besides, Hidden Markov Model (HMM) is also widely used in the speech recognition area. However, the performance of HMM model is limited, and not suitable for long speeches with various tunes. Thus, only the deep learning method is applied in this project.

## How to run the code

First, you have to go to the directory of this project

```shell
cd <project_path>
```

In the project path, install the setup.py

```shell
pip install -e .
```

Next, go to the web folder

```shell
cd web
```

Finally, run the web_app.py

```Shell
python web_app.py
```

The application will be activated in localhost.

