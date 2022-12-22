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
import functions as fn
from dtreeviz.trees import dtreeviz 
from dtreeviz.models.shadow_decision_tree import ShadowDecTree
from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree

class variables:
    counter = 0


model_speeker = pickle.load(open('M.pickle', 'rb'))
model_speech = pickle.load(open('Classifspeech.pickle', 'rb'))



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
  
    first = signal[:int(sample_rate*15)]
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(
        first, Fs=sample_rate)
    img = image(fig, "result")
    return img



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
    name = "test"
    files = os.path.join(os.path.abspath('Sound recordings/{}').format(name)+ ('\\') +str(files['file']))
    audio_features =  fn.extract_features(files,name="test")
    speech_features = fn.extract_speech_features(files,name="test")
    connect_audio_features = fn.feat(audio_features)
    connect_speech_features = fn.feat(speech_features)

    filelist = os.listdir('Sound recordings//test//')

    df_test = pd.DataFrame(filelist)
    df_test['label']=0
    df_test = df_test.rename(columns={0:'file'})
    features_label2 = df_test.apply(fn.extract_features, axis=1)
    features_speaker=fn.feat(features_label2)
    prediction_speaker = model_speeker.predict(features_speaker)

    filelist_sp = os.listdir('Sound recordings2//test//test.wav')
    df_test_sp = pd.DataFrame(filelist_sp)
    df_test_sp['label']=1
    df_test_sp =df_test_sp.rename(columns={0:'file'})
    features_label3 = df_test_sp.apply(fn.extract_speech_features, axis=1)
    features_speech=fn.feat(features_label3)
    prediction_speech = model_speech.predict(features_speech)
    features_label3 = df_test.apply(fn.extract_speech_features, axis=1)
    features_speech=fn.feat(features_label3)
    prediction_speech = model_speech.predict(features_speech)
    # fn.draw_path(model_speeker,features_speaker)

    prediction_speaker = fn.predict_speaker( prediction_speaker)
    prediction_speech= fn.predict_spesch( prediction_speech)


    pickle.dump(audio_features , open('x.pkl', 'wb'))
    pickle.dump(df_test['label'], open('y.pkl', 'wb'))
    def get_path_as_Histogram(features):
        X = pickle.load(open('x.pkl','rb'))
        y = pickle.load(open('y.pkl','rb'))
        clf = pickle.load(open('Clll.pickle','rb'))
        feature_names = ['mfccs', 'chroma', 'mel', 'contrast', 'tonnetz']*len(X)
        target_names = ["others", 'Bassant','Turky', 'Mayar', 'Ereny']
        viz = dtreeviz(clf,
                    np.array(X), 
                    np.array(y),
                    target_name="Member",
                    feature_names = feature_names,
                    class_names= target_names, 
                    title="SBME3 voice recognition",
                    fontname="Arial",
                    scale=1.5,
                    X=features[0],
                    fancy=False)
        filename = 'static/assets/image/Histogramtree.svg'
        viz.save(filename)
        return filename
   
def draw_path(model_speeker,features_speaker):
       # clf=sklearn.model_selection 
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
   
   samples =  features_speaker[0]
   print(samples)
   decision_paths = model_speeker.decision_path(samples.reshape(1,-1))
   threshold=features_speaker[0][4]
   for decision_path in decision_paths:
      for n, node_value in enumerate(decision_path.toarray()[0]):
         if node_value == 0:
               continue
         node = graph.get_node(str(n))[0]            
         node.set_fillcolor('yellow')
         labels = node.get_attributes()['label'].split('<br/>')
         for i, label in enumerate(labels):
               if label.startswith('samples = '):
                  labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)
   
         node.set('label', '<br/>'.join(labels))
   
#    filename = 'static/assets/images/tree.png'
    
    spectrum = visualize("Sound recordings/test/test.wav")
    spectrum = 'static/assets/images/tree.png'
    graph.write_png(spectrum)




    circles_tree =  'static/assets/images/tree_path.svg'
    dist_fig = tonnetz_feature(
        "Sound recordings/test/test.wav", features_speech)
    dist_fig = 'static/assets/images/tree_dist.svg'
    # 'static/assets/images/tonnetz'+str(variables.counter)+'.jpg'
#
    dot_data = tree.export_graphviz(model_speeker, out_file=None,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

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


    return render_template("index.html", prediction_text='{}'.format(prediction_speaker), speech_text='{}'.format(prediction_speech), words='{}'.format(text), img=spectrum, tonnetz_fig=dist_fig, img2=circles_tree)


if __name__ == '__main__':
    app.run(debug=True, port=8000)