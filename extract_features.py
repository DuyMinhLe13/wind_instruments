import numpy as np
def FFT(x):
    """
    args:
    x: np.array of audio
    input should have a length of
    power of 2.
    """
    N = len(x)

    if N == 1:
        return x
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N)

        X = np.concatenate(\
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])
        return X

def zero_crossing_rate(x, samplerate=44100):
    """
    args:
    x: np.array of audio
    samplerate: sample rate of audio
    """
    return int(np.diff(x > 0).sum())

def avg_energy(x, samplerate=44100):
    """
    args:
    x: np.array of audio
    samplerate: sample rate of audio
    """
    return float((x ** 2).sum() / x.shape[0])

def bandwidth(x, samplerate=44100, threshold=-41):
    """
    args:
    x: np.array of audio
    samplerate: sample rate of audio
    threshold: 3db above silent level
    """
    length = x.shape[0]
    magnitudes = 10 * np.log10(np.abs(FFT(x))[:length//2+1] / float(length)) # magnitudes of positive frequencies
    freqs = np.linspace(0, samplerate/2, length//2+1) # positive frequencies
    positive_mag_index = (magnitudes > threshold).nonzero()[0]
    return float(freqs[positive_mag_index[-1]] - freqs[positive_mag_index[0]])

def spectral_centroid(x, samplerate=44100):
    """
    args:
    x: np.array of audio
    samplerate: sample rate of audio
    """
    length = x.shape[0]
    magnitudes = np.abs(FFT(x))[:length//2+1] # magnitudes of positive frequencies
    freqs = np.linspace(0, samplerate/2, length//2+1) # positive frequencies
    return float(np.sum(magnitudes*freqs) / np.sum(magnitudes)) # return weighted mean

def pitch_frequency(x, samplerate=44100):
    """
    args:
    x: np.array of audio
    samplerate: sample rate of audio
    """
    length = x.shape[0]
    magnitudes = np.abs(FFT(x))[:length//2+1]
    freqs = np.linspace(0, samplerate/2, length//2+1)
    return float(freqs[np.argmax(magnitudes)])

def extract_features(y, sr=44100, window_size=8192, stride=4096, max_len=131072):
    if max_len != None: y = y[:max_len]
    windows = [y[i:i+window_size] for i in range(0, y.shape[0]-window_size, stride)]
    zero_crossing_rates = []
    avg_energies = []
    bandwidths = []
    spectral_centroids = []
    pitch_frequencies = []
    for window in windows:
        zero_crossing_rates.append(zero_crossing_rate(window, sr))
        avg_energies.append(avg_energy(window, sr))
        bandwidths.append(bandwidth(window, sr))
        spectral_centroids.append(spectral_centroid(window, sr))
        pitch_frequencies.append(pitch_frequency(window, sr))
    return zero_crossing_rates, avg_energies, bandwidths, spectral_centroids, pitch_frequencies

