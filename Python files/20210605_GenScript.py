import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sp

sample_rate = 20*1e3  # in Hertz [Hz]
recording_time = 2  # in seconds [s]
periodogram_resolution = 0.25  # in seconds [s]

acer_pc = 1  # 1: true/0: false

# Recording audio
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
if acer_pc is 1:
    upper_pow2_length = np.ceil(np.log2(total_samples))
    fft_length = int(2 ** upper_pow2_length)
else:
    fft_length = total_samples

spectrum_rec_audio_1 = sp.fftshift(sp.fft(rec_audio[:, 0], fft_length))
spectrum_rec_audio_2 = sp.fftshift(sp.fft(rec_audio[:, 1], fft_length))

power_spectrum_1 = np.abs(spectrum_rec_audio_1 / fft_length) ** 2
power_spectrum_2 = np.abs(spectrum_rec_audio_2 / fft_length) ** 2

power_spectrum_1_dB = 10 * np.log10(power_spectrum_1)
power_spectrum_2_dB = 10 * np.log10(power_spectrum_2)

freq_vector_pre = np.linspace(-sample_rate/2, sample_rate/2, fft_length+1)
freq_vector = freq_vector_pre[0:-1]

plt.plot(freq_vector, power_spectrum_1_dB)
plt.plot(freq_vector, power_spectrum_2_dB)
plt.xlim([0, 10e3])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power [dB]')
plt.show()

# Periodogram
samples_per_PG_resolution = int(periodogram_resolution/(1/sample_rate))
PG_slices = int(total_samples/samples_per_PG_resolution)

PG_upper_pow2_length = np.ceil(np.log2(samples_per_PG_resolution))
PG_fft_length = int(2 ** PG_upper_pow2_length)

reshaped_rec_audio_1 = np.reshape(rec_audio[:, 0], [PG_slices, samples_per_PG_resolution])
reshaped_rec_audio_2 = np.reshape(rec_audio[:, 1], [PG_slices, samples_per_PG_resolution])

power_spectrum_PG_1 = np.zeros([PG_slices, PG_fft_length])

for ind_1 in range(0, PG_slices):
    spectrum_rec_audio_local_1 = sp.fftshift(sp.fft(reshaped_rec_audio_1[ind_1, :], PG_fft_length))
    power_spectrum_1_local = np.abs(spectrum_rec_audio_local_1 / PG_fft_length) ** 2
    power_spectrum_1_dB_local = 10 * np.log10(power_spectrum_1_local)
    power_spectrum_PG_1[ind_1, :] = power_spectrum_1_dB_local

plt.imshow(np.transpose(power_spectrum_PG_1), aspect='auto', interpolation='none', origin='lower')
plt.show()
