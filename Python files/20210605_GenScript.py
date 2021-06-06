import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sp


def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]


sample_rate = 20*1e3  # in Hertz [Hz]
recording_time = 120  # in seconds [s]
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

# upper_pow2_length = np.ceil(np.log2(total_samples))
# fft_length = int(2 ** upper_pow2_length)

fft_length = total_samples

spectrum_rec_audio_1 = sp.fftshift(sp.fft(rec_audio[:, 0], fft_length))
spectrum_rec_audio_2 = sp.fftshift(sp.fft(rec_audio[:, 1], fft_length))

power_spectrum_1 = np.abs(spectrum_rec_audio_1 / fft_length) ** 2
power_spectrum_2 = np.abs(spectrum_rec_audio_2 / fft_length) ** 2

power_spectrum_1_dB = 10 * np.log10(power_spectrum_1)
power_spectrum_2_dB = 10 * np.log10(power_spectrum_2)

freq_vector_pre = np.linspace(-sample_rate/2, sample_rate/2, fft_length+1)
freq_vector = freq_vector_pre[0:-1]

try:
    plt.plot(freq_vector, power_spectrum_1_dB)
    plt.plot(freq_vector, power_spectrum_2_dB)
    plt.xlim([0, sample_rate/4])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power [dB]')
    plt.show()
except:
    print('Spectrum not available')

# Periodogram
samples_per_PG_resolution = int(periodogram_resolution/(1/sample_rate))
PG_slices = int(total_samples/samples_per_PG_resolution)

PG_upper_pow2_length = np.ceil(np.log2(samples_per_PG_resolution))
PG_fft_length = int(2 ** PG_upper_pow2_length)

PG_freq_vector_pre = np.linspace(-sample_rate/2, sample_rate/2, PG_fft_length+1)
PG_freq_vector = PG_freq_vector_pre[0:-1]
PG_freq_vector_pos = PG_freq_vector[PG_freq_vector >= 0]

PG_time_vector_pre = np.linspace(0, PG_slices * periodogram_resolution, PG_slices + 1)
PG_time_vector = PG_time_vector_pre[1:-1]

reshaped_rec_audio_1 = np.reshape(rec_audio[:, 0], [PG_slices, samples_per_PG_resolution])
reshaped_rec_audio_2 = np.reshape(rec_audio[:, 1], [PG_slices, samples_per_PG_resolution])

power_spectrum_PG_1 = np.zeros([PG_slices, int(PG_fft_length/2)])
print(np.shape(power_spectrum_PG_1))

for ind_1 in range(0, PG_slices):
    spectrum_rec_audio_local_1 = sp.fftshift(sp.fft(reshaped_rec_audio_1[ind_1, :], PG_fft_length))
    power_spectrum_1_local = np.abs(spectrum_rec_audio_local_1 / PG_fft_length) ** 2
    power_spectrum_1_dB_local = 10 * np.log10(power_spectrum_1_local)
    power_spectrum_PG_1[ind_1, :] = power_spectrum_1_dB_local[PG_freq_vector >= 0]

try:
    plt.imshow(np.transpose(power_spectrum_PG_1), aspect='auto', interpolation='none',
               extent=extents(PG_time_vector) + extents(PG_freq_vector_pos), origin='lower')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.show()
except:
    print('Periodogram not available')
