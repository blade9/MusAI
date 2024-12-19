from scipy.io import wavfile



sr, y = wavfile.read("Data/angel_beats.wav")
print(f"Sample rate: {sr}, Data shape: {y.shape}")
