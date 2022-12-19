from flask import Flask, render_template, request
import numpy as np
import pickle
import wave
import os
import shutil
import librosa
import pandas as pd
import speech_recognition as sr
import librosa.display
from io import BytesIO
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import pydotplus
from sklearn import tree


class variables:
    counter = 0


model_speeker = pickle.load(open('M.pickle', 'rb'))
model_speech = pickle.load(open('Classifspeech.pickle', 'rb'))
# Classifi.pickle iii iiiii


def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.
    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
                successful
    "error":   `None` if no error occured, otherwise a string containing
                an error message if the API could not be reached or
                speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
                otherwise a string containing the transcribed text
    """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        with open('Sound recordings/test/test.wav', 'wb') as f:
            f.write(audio.get_wav_data())
            shutil.copy2('Sound recordings/test/test.wav',
                         'Sound recordings2/test/test.wav')

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response


def image(fig, name):
    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    # Embed the result in the html output.
    data = base64.b64encode(img.getbuffer()).decode("ascii")
    image_file_name = 'static/assets/images/' + \
        str(name)+str(variables.counter)+'.jpg'
    plt.savefig(image_file_name)
    return f"<img src='data:image/png;base64,{data}'/>"


def visualize(file_name):
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.xlabel("Time (Sec)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    ax = sns.set_style(style='darkgrid')
    sample_rate, signal = wavfile.read(file_name)
    # select left channel only
    # signal = signal[:,0]
    # trim the first 125 seconds
    first = signal[:int(sample_rate*15)]
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(
        first, Fs=sample_rate)
    img = image(fig, "result")
    return img


# def mfcc(audio,features):
#     signal, sample_rate = librosa.load(audio)
#     mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40)
#     S = librosa.feature.melspectrogram(
#         y=signal, sr=sample_rate, n_mels=128, fmax=8000)
#     fig, ax = plt.subplots()
#     plt.xlabel
#     img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
#     plt.axhline(y = features[0][1], color = 'r', linestyle = '-')
#     fig.colorbar(img, ax=ax)
#     ax.set(title='MFCC')
#     img = image(fig, "mfcc")
#     return img


def tonnetz_feature(audio, features):
    signal, sample_rate = librosa.load(audio)
    signal = librosa.effects.harmonic(signal)
    tonnetz = librosa.feature.tonnetz(y=signal, sr=sample_rate)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(tonnetz, y_axis='log', x_axis='time',ax=ax)
    plt.axhline(y=(features[0][4]), color='r', linestyle='-')
    ax.set(title='Tonal Centroids (Tonnetz)')
    ax.label_outer()
    fig.colorbar(img, ax=ax)
    img = image(fig, "tonnetz")
    return img


app = Flask(__name__, template_folder="templates")

ButtonPressed = 0


@app.route('/', methods=['GET', 'POST'])
def record():
    # create recognizer and mic instances
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    guess = recognize_speech_from_mic(recognizer, microphone)

    def extract_features(files, name="test"):
        try:
            # Sets the name to be the path to where the file is in my computer
            file_name = os.path.join(os.path.abspath(
                'Sound recordings/{}').format(name) + ('\\') + str(files['file']))

            # Loads the audio file as a floating point time series and assigns the default sample rate
            # Sample rate is set to 22050 by default

            X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

            # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

            # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
            stft = np.abs(librosa.stft(X))

            # Computes a chromagram from a waveform or power spectrogram.
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)

            # Computes a mel-scaled spectrogram.
            mel = np.mean(librosa.feature.melspectrogram(
                X, sr=sample_rate).T, axis=0)

            # Computes spectral contrast
            contrast = np.mean(librosa.feature.spectral_contrast(
                S=stft, sr=sample_rate).T, axis=0)

            # Computes the tonal centroid features (tonnetz)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                                      sr=sample_rate).T, axis=0)

            # We add also the classes of each file as a label at the end
            label = files.label

        except:
            print(files)

        return mfccs, chroma, mel, contrast, tonnetz, label

    def extract_speech_features(files, name="test"):
     #     try:
        # Sets the name to be the path to where the file is in my computer
        file_name = os.path.join(os.path.abspath(
            'Sound recordings2/{}').format(name) + ('\\') + str(files['file']))

        # Loads the audio file as a floating point time series and assigns the default sample rate
        # Sample rate is set to 22050 by default

        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        HOP_LENGTH = 512
        FRAME_SIZE = 1024
        mfccs = np.mean(librosa.feature.mfcc(
            y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        # Computes spectral centroied
        sc = np.mean(librosa.feature.spectral_centroid(
            y=X, sr=sample_rate, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH).T, axis=0)
        # Computes spectral bandwidth
        spectral_bw = np.mean(librosa.feature.spectral_bandwidth(
            y=X, sr=sample_rate, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH).T, axis=0)
        # Computes RMSE
        rms = np.mean(librosa.feature.rms(
            X, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH).T, axis=0)
        # Computes Zero-crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(
            X, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH).T, axis=0)
        # Computes rolloff
        rolloff = np.mean(librosa.feature.spectral_rolloff(
            X, sr=sample_rate).T, axis=0)
        # We add also the classes of each file as a label at the end
        label = files.label

        return mfccs, sc, spectral_bw, zcr, rolloff,  label
        # show the user the transcription
    # print("You said: {}".format(guess["transcription"]))

    def feat(features_label):
        features = []
        for i in range(0, len(features_label)):
            features.append(np.concatenate((features_label[i][0], features_label[i][1],
                                            features_label[i][2], features_label[i][3],
                                            features_label[i][4]), axis=0))
        return features
# read them into pandas
    filelist = os.listdir('Sound recordings//test//')

    df_test = pd.DataFrame(filelist)
    df_test['label'] = 0
    df_test = df_test.rename(columns={0: 'file'})
    features_label2 = df_test.apply(extract_features, axis=1)
    features_speaker = feat(features_label2)
    prediction_speaker = model_speeker.predict(features_speaker)

    filelist_sp = os.listdir('Sound recordings2//test//')
    df_test_sp = pd.DataFrame(filelist_sp)
    df_test_sp['label'] = 1
    df_test_sp = df_test_sp.rename(columns={0: 'file'})
    features_label3 = df_test_sp.apply(extract_speech_features, axis=1)
    features_speech = feat(features_label3)
    prediction_speech = model_speech.predict(features_speech)
    features_label3 = df_test.apply(extract_speech_features, axis=1)
    features_speech = feat(features_label3)
    prediction_speech = model_speech.predict(features_speech)
    text = ' '
    if prediction_speaker == ['0']:
        prediction_speaker = "Other"
    elif prediction_speaker == ['1']:
        prediction_speaker = "Bassent"
    elif prediction_speaker == ['2']:
        prediction_speaker = "Turky"
    elif prediction_speaker == ['3']:
        prediction_speaker = "Mayar"
    elif prediction_speaker == ['4']:
        prediction_speaker = "Ereny"
    result = prediction_speaker == "Bassent" or prediction_speaker == "Turky" or prediction_speaker == "Mayar" or prediction_speaker == "Ereny"
    if prediction_speech == ['5']:
        text = 'Open the door'
        if result:
            prediction_speech = "Verified"
        else:
            prediction_speech = "Not Verified"
   #  elif prediction_speech == ['5'] :
   #          prediction_speech = "Verified"
   #          text = 'Open the door'
    elif prediction_speech == ['6']:
        prediction_speech = "Not allowed"
        text = 'cannot define'
    print(prediction_speaker)
    print(prediction_speech)
    spectrum = visualize("Sound recordings/test/test.wav")
    spectrum = 'static/assets/images/result'+str(variables.counter)+'.jpg'
    variables.counter += 1

    # mfcc_fig = mfcc("Sound recordings/test/test.wav", features_speech)
    # mfcc_fig = 'static/assets/images/mfcc'+str(variables.counter)+'.jpg'
    tonnetz_fig = tonnetz_feature(
        "Sound recordings/test/test.wav", features_speech)
    tonnetz_fig = 'static/assets/images/tonnetz'+str(variables.counter)+'.jpg'
#
    dot_data = tree.export_graphviz(model_speeker, out_file=None,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    # empty all nodes, i.e.set color to white and number of samples to zero
    for node in graph.get_node_list():
        if node.get_attributes().get('label') is None:
            continue
        if 'samples = ' in node.get_attributes()['label']:
            labels = node.get_attributes()['label'].split('<br/>')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = 0'
            node.set('label', '<br/>'.join(labels))
            node.set_fillcolor('white')

    samples = features_speaker[0]
    
    decision_paths = model_speeker.decision_path(samples.reshape(1, -1))

    for decision_path in decision_paths:
        for n, node_value in enumerate(decision_path.toarray()[0]):
            if node_value == 0:
                continue
            node = graph.get_node(str(n))[0]
            node.set_fillcolor('yellow')
            labels = node.get_attributes()['label'].split('<br/>')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = {}'.format(
                        int(label.split('=')[1]) + 1)

            node.set('label', '<br/>'.join(labels))

    filename = 'static/assets/images/tree.png'
    graph.write_png(filename)
# #
#     from dtreeviz.trees import dtreeviz # remember to load the package

#     viz = dtreeviz(model_speeker, X, y,
#                     target_name="target",
#                     feature_names= features_speaker,
#                     class_names=list(features_speaker.target_names))

#     viz
    # return render_template("index.html", words = guess["transcription"],prediction_text=' The speaker is:{}'.format(prediction_speaker),speech_text=' The speech is:{}'.format(prediction_speech))
    return render_template("index.html", prediction_text='{}'.format(prediction_speaker), speech_text='{}'.format(prediction_speech), words='{}'.format(text), img=spectrum, tonnetz_fig=tonnetz_fig)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
