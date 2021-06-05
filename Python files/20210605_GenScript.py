import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

sample_rate = 40*1e3  # in Hertz [Hz]
recording_time = 5  # in seconds [s]

rec_audio = sd.rec(int(sample_rate * recording_time), sample_rate, 2)
sd.wait()

# Time domain representation of the recorded audio
total_samples = len(rec_audio)
time_vector_pre = np.linspace(0, recording_time, total_samples+1)
time_vector = time_vector_pre[0:-1]

plt.plot(time_vector, rec_audio)
plt.show()
