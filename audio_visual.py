import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import librosa.display
import matplotlib.image as img
import streamlit as st 
from pathlib import Path

st.set_option('deprecation.showPyplotGlobalUse', False)

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def load_audio(path):

    #load audio
    y, sr = librosa.load(path)

    return y, sr

def short_term_fourier(y, sr):
    '''takes a path of an audio file and returns the mel spectrogram in relation to Hz and key'''
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                    fmax=8000)

    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    
    return fig, ax
    
def chromogram(y, sr):
    '''takes an audio file and returns a chromogram of the audio file'''
    
    # extracting the chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # plot the audio
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    fig.colorbar(img, ax=ax)
    ax.set(title='Chromagram explicitly in Eb:maj')
    
    return fig, ax
    
def mfcc(y, sr):
    '''returns the mel frequency cepstral coefficients of an audio file'''
 
    # computing the mfcc
    m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)
    
    # plot the mfcc
    fig, ax = plt.subplots() 
    img = librosa.display.specshow(m_htk, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC dctype 3')
    
    return fig, ax

def rms(y, sr):
    '''returns the rms of an audio file'''

    # computing the rss
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)

    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(rms)
    ax[0].semilogy(times, rms[0], label='RMS Energy')
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='log Power spectrogram')
    

    return fig, ax


def spec_cent(y, sr):
    '''calculates the spectral centroids'''

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    S, phase = librosa.magphase(librosa.stft(y=y))

    times = librosa.times_like(cent)
    fig, ax = plt.subplots()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax)
    ax.plot(times, cent.T, label='Spectral centroid', color='w')
    ax.legend(loc='upper right')
    ax.set(title='log Power spectrogram')

    return fig, ax

def spec_band(y, sr):
    '''calculates the spectral bandwidths'''

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    S, phase = librosa.magphase(librosa.stft(y=y))

    freqs, times, D = librosa.reassigned_spectrogram(y, fill_nan=True)

    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(spec_bw)
    centroid = librosa.feature.spectral_centroid(S=S)
    ax[0].semilogy(times, spec_bw[0], label='Spectral bandwidth')
    ax[0].set(ylabel='Hz', xticks=[], xlim=[times.min(), times.max()])
    ax[0].legend()
    ax[0].label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='log Power spectrogram')
    ax[1].fill_between(times, np.maximum(0, centroid[0] - spec_bw[0]),
                    np.minimum(centroid[0] + spec_bw[0], sr/2),
                    alpha=0.5, label='Centroid +- bandwidth')
    ax[1].plot(times, centroid[0], label='Spectral centroid', color='w')
    ax[1].legend(loc='lower right')

    return fig, ax

st.title("Audio-Visualizer")
st.header("An application that accepts a given audio and displays different visual representations of the audio.")
audio = st.file_uploader(label = "Upload your audio snippet :)", type = "wav")

if audio is not None:
    y, sr = load_audio(audio)
    fig_st, ax_st = short_term_fourier(y, sr)
    fig_chrom, ax_chrom  = chromogram(y, sr)
    fig_mfcc, ax_mfcc = mfcc(y,sr)
    fig_rms = rms(y, sr)
    fig_spec, ax_spec = spec_cent(y, sr)
    fig_spec_band, ax_spec_band = spec_band(y, sr)


    st.audio(audio)
    st.subheader("How can we analyze audio visually?")

    st.caption("MEL-Frequency Spectogram")
    st.pyplot(fig_st)
    st.markdown("The Mel-Spectrogram is used to visually present a signal's amplitude (loudness), as it varies at different frequencies. The frequencies is on Mel scale, which is a logarithmic transformation of a signal's frequency. Since it is harder for humans to see the difference between higher frequencies (1000 and 1100 Hz) than between lower frequencies (100 and 200 Hz), Mel enables sounds of equal distance on its scale is perceived to be also equal distance to humans.")
    st.caption("Chromogram")
    st.pyplot(fig_chrom)
    st.markdown("The chromogram or chroma feature is a visual representation of the twelve different pitch classes assuming an equal-tempered scale (C, C♯, D, D♯, E , F, F♯, G, G♯, A, A♯, B). It aggregates over a given time window and displays the pitch content of each pitch class. A chromogram can also be referred to as a pitch class profile as it enables the viewer to quickly gain an understanding of the key played or even the chord structures used in a song. ")
    st.caption("MEL Frequency Cepstral Coefficients")
    st.pyplot(fig_mfcc)
    st.markdown(
        """
        The main aspect when visually analyzing audio, is to understand that it is never bound to a specific point in time but always only exists between point A and B in time. Similarly to analyzing videos, we can analyze different frames of audio at a point in time and observe the change of overlapping frames. 
        1. Pre-Emphasis: We take the normal waveform of the audio and normalize the magnitudes of higher and lower frequencies to balance them out 
        2. Framing & Windowing: We split the signal into small, overlapping frames and apply a window function to them (calculating a Mel Spectogram over the whole audio leads to a loss of frequency content)
        3. Short Term Fourier Transform: We now apply Short-Term Fourier Transformations on each window (Fourier Transform =  converts the signal from the time domain into the frequency domain.) → Spectrum 
        4. Apply Log-Amplitude & Mel-Scaling: Based on Filter banks we can extract the frequency bands and represent the non-linear perception of the human ear (Power to dB) → Log-Power Spectrum 
        5. Discrete Cosine Transform: Since the coefficients of the filter banks might be highly correlated, we apply the Discrete Cosine Transformation to build a compressed view → Cepstrum (basically a compressed version representing the change of a MEL spectogram for each frame)
        """)
    st.caption("Spectral Centroids")
    st.pyplot(fig_spec)
    st.markdown("We use spectral features to observe how the frequency of a sound signal is changing and can therefore represent the center of mass of a spectrum. It is basically a weighted median of the spectrum that can be used to understand the brightness of an audio file. By doing so, we can distinguish between two different instruments that play the same key, since the tone quality will differ, which is represented in the spectral centroid. Similarly, the spectral bandwidth measures the difference between the higher and lower frequencies. If the centroid is the center of the spectrum, then the bandwidth is the sum of maximum deviation on both sides. ")
    st.caption("Spectral Bandwidhts")
    st.pyplot(fig_spec_band)
    #st.caption("RMS")
    #st.pyplot(fig_rms)
    
 
