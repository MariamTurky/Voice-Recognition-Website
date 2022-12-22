import os
import shutil
import librosa
import numpy as np
import pydotplus
from sklearn import tree

def extract_features(files,name="test"):
      try:
        # Sets the name to be the path to where the file is in my computer
        file_name = os.path.join(os.path.abspath('Sound recordings/{}').format(name)+ ('\\') +str(files['file']))

        # Loads the audio file as a floating point time series and assigns the default sample rate
        # Sample rate is set to 22050 by default
        
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        
        # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

        # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
        stft = np.abs(librosa.stft(X))

        # Computes a chromagram from a waveform or power spectrogram.
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

        # Computes a mel-scaled spectrogram.
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

        # Computes spectral contrast
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

        # Computes the tonal centroid features (tonnetz)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
        sr=sample_rate).T,axis=0)

        # We add also the classes of each file as a label at the end
        label = files.label
        
      except:
        print(files)

      return mfccs, chroma, mel, contrast, tonnetz, label

def extract_speech_features(files,name="test"):
    #     try:
        # Sets the name to be the path to where the file is in my computer
        file_name = os.path.join(os.path.abspath('Sound recordings2/{}').format(name)+ ('\\') +str(files['file']))

        # Loads the audio file as a floating point time series and assigns the default sample rate
        # Sample rate is set to 22050 by default
        
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        HOP_LENGTH = 512
        FRAME_SIZE = 1024
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
        # Computes spectral centroied
        sc = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH).T,axis=0)
        # Computes spectral bandwidth
        spectral_bw = np.mean(librosa.feature.spectral_bandwidth(y=X, sr=sample_rate, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH).T,axis=0)
        #Computes RMSE
        rms = np.mean(librosa.feature.rms(X, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH).T,axis=0)
        #Computes Zero-crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(X, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH).T,axis=0)
        #Computes rolloff
        rolloff = np.mean(librosa.feature.spectral_rolloff(X, sr=sample_rate ).T,axis=0)
        # We add also the classes of each file as a label at the end
        label = files.label
        
        
        return mfccs,sc , spectral_bw,zcr,rolloff,  label
      # show the user the transcription
   # print("You said: {}".format(guess["transcription"]))
def feat(features_label):
         features = []
         for i in range(0, len(features_label)):
                features.append(np.concatenate((features_label[i][0], features_label[i][1], 
                features_label[i][2], features_label[i][3],
                features_label[i][4]), axis=0))
         return features  
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
   
   filename = 'static/assets/images/tree.png'
   graph.write_png(filename)


def predict_speech(prediction_speech,prediction_speaker):
   result = prediction_speaker(prediction_speaker) == "Bassent" or prediction_speaker(( prediction_speaker)) == "Turky" or prediction_speaker(( prediction_speaker)) == "Mayar" or prediction_speaker(( prediction_speaker)) == "Ereny"
   if prediction_speech == ['5']:
        text = 'Open the door'
        if result:
            prediction_speech = "Verified"
        else:
            prediction_speech = "Not Verified"
            prediction_speech = "Verified"
            text = 'Open the door'
   elif prediction_speech == ['6']:
         prediction_speech= "Not allowed"
         text='cannot define'
   return text

def predict_speaker( prediction_speaker) :
    if prediction_speaker == ['0']:
            prediction_speaker = "Other"
    elif prediction_speaker == ['1']:
            prediction_speaker = "Bassent"
    elif prediction_speaker == ['2']:
            prediction_speaker = "Turky"
    elif prediction_speaker == ['3']:
            prediction_speaker  = "Mayar"
    elif prediction_speaker == ['4']:
            prediction_speaker  = "Ereny"
    return prediction_speaker

