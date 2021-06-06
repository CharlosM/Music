import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sp

sample_rate = 40*1e3  # in Hertz [Hz]
recording_time = 0.5  # in seconds [s]

run_plot = 1
ind = 0

while run_plot is 1:

    rec_audio = sd.rec(int(sample_rate * recording_time), sample_rate, 2)
    sd.wait()

    # Time domain representation of the recorded audio
    total_samples = len(rec_audio)

    # Frequency domain representation of the recorded audio

    spectrum_rec_audio_1 = sp.fftshift(sp.fft(rec_audio[:, 0]))
    spectrum_rec_audio_2 = sp.fftshift(sp.fft(rec_audio[:, 1]))

    power_spectrum_1 = np.abs(spectrum_rec_audio_1 / total_samples) ** 2
    power_spectrum_2 = np.abs(spectrum_rec_audio_2 / total_samples) ** 2

    power_spectrum_1_dB = 10 * np.log10(power_spectrum_1)
    power_spectrum_2_dB = 10 * np.log10(power_spectrum_2)

    freq_vector_pre = np.linspace(-sample_rate / 2, sample_rate / 2, total_samples + 1)
    freq_vector = freq_vector_pre[0:-1]

    ind = ind + 1
    print(ind)

    plt.plot(freq_vector, power_spectrum_1_dB)
    plt.axis([0, sample_rate / 2, -110, 0])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power [dB]')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()
