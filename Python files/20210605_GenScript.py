import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sp

sample_rate = 40*1e3  # in Hertz [Hz]
recording_time = 60  # in seconds [s]

rec_audio = sd.rec(int(sample_rate * recording_time), sample_rate, 2)
sd.wait()

# Time domain representation of the recorded audio
total_samples = len(rec_audio)
time_vector_pre = np.linspace(0, recording_time, total_samples+1)
time_vector = time_vector_pre[0:-1]

plt.plot(time_vector, rec_audio)
plt.xlim([0, time_vector[-1]])
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [a.u.]')
plt.show()

# Frequency domain representation of the recorded audio
#upper_pow2_length = np.ceil(np.log2(total_samples))
#fft_length = int(2 ** upper_pow2_length)

spectrum_rec_audio_1 = sp.fftshift(sp.fft(rec_audio[:, 0]))
spectrum_rec_audio_2 = sp.fftshift(sp.fft(rec_audio[:, 1]))

power_spectrum_1 = np.abs(spectrum_rec_audio_1 / total_samples) ** 2
power_spectrum_2 = np.abs(spectrum_rec_audio_2 / total_samples) ** 2

power_spectrum_1_dB = 10 * np.log10(power_spectrum_1)
power_spectrum_2_dB = 10 * np.log10(power_spectrum_2)

freq_vector_pre = np.linspace(-sample_rate/2, sample_rate/2, total_samples+1)
freq_vector = freq_vector_pre[0:-1]

plt.plot(freq_vector, power_spectrum_1_dB)
plt.plot(freq_vector, power_spectrum_2_dB)
plt.xlim([0, freq_vector[-1]])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power [dB]')
plt.show()

