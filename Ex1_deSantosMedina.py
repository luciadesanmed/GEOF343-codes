import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis

#By: Ana Lucia de Santos Medina

#The file named ‘miros 2016 02 02 00 00.asc’ contains sea level measurements from the North Sea (Ekofisk) made
#with a Miros radar. The record is 20 minutes long and the sampling rate was 2 Hz (one measurement each 0.5 s).
#The date of the record is 2 February 2016, 00 UTC, when there was a storm.

#a) Plot the time series and describe the data. What are the highest wave frequencies that we can resolve in this
#timeseries?

#Opening the file:
file = "miros_2016_02_02_00_00.asc"
df = pd.read_csv(file, header=None)
sea_level = df[0].to_numpy()

#Defining time vector:
dt = 0.5 #in seconds
time_sec = np.arange(len(sea_level)) * dt
time_min = time_sec / 60

#Timeseries plot:
plt.figure(figsize=(10, 4))
plt.plot(time_min, sea_level, color="black", linewidth=1)
plt.ylabel('Sea level [m]')
plt.xlabel('Time [min]')
plt.xlim(0, 20)
plt.title('Sea level timeseries (starting 02.02.2016, 00 UTC)')
plt.tight_layout()
plt.savefig('a_sea_level_timeseries.png', dpi=300)
plt.close()

#The highest wave frequencies that we can resolve in this timeseries are given by the Nyquist frequency (f_N):
f_N = 1/(2*dt)

print(f'The highest frequency we can resolve in this timeseries is {f_N:.2f} Hz.')

#b) Find the mean value, and compute the sea surface elevation eta = eta(t) (= mean value minus the data).

#Finding the mean value:
sealvl_mean = np.mean(sea_level)
print(f'The sea level mean value is {sealvl_mean:.3f} m.')

#Compute sea surface elevation:
eta = sealvl_mean - sea_level

#c) Calculate the significant wave height Hm0:

#We will obtain Hm0 following Hm0 = 4*sqrt(m0) = 4*sqrt((var(eta))), where m0 is the zeroth order moment of the variance density spectrum
#We will obtain Hm0 using 2 methods: calculating through the variance of eta and m0.
#For m0 we will obtain the variance density spectrum by performing the Fast Fourier Transform (FFT) of the sea surface elevation

Hm0_from_std = 4*np.std(eta)

print(f'The value of Hm0 is {Hm0_from_std:.2f} m (using std(eta)).')

#Number of records per second:
sam_rate = 2 #sampling rate
N = len(eta) #total number of records

freq = fftfreq(N, 1/sam_rate) #frequency vector

pos_freq = freq > 0
freq_plot = freq[pos_freq]

fft_raw = fft(eta)

#Only the positive side of the spectrum:
spec = (2.0/(sam_rate*N)) * np.abs(fft_raw[pos_freq])**2

#Calculating m0 by integrating the spectrum:
df_freq = sam_rate/N
m0 = np.sum(spec) * df_freq

Hm0_from_spec = 4*np.sqrt(m0)

print(f'The value of Hm0 is {Hm0_from_spec:.2f} m (using spectrum).')

#d) Plot the whole 20 minutes of the sea level time record, and describe the sea state.
plt.figure(figsize=(10, 4))
plt.plot(time_min, eta, color="black", linewidth=1)
plt.axhline(y=0, color='red')
plt.ylabel('Sea surface elevation [m]')
plt.xlabel('Time [min]')
plt.xlim(0, 20)
plt.title('Sea surface elevation timeseries (starting 02.02.2016, 00 UTC)')
plt.tight_layout()
plt.savefig('d_sea_surface_elevation_20min.png', dpi=300)
plt.close()

#e) Plot the first three minutes of the sea level to see some details clearer.
plt.figure(figsize=(10, 4))
plt.plot(time_min, eta, color="black", linewidth=1)
plt.axhline(y=0, color='red')
plt.ylabel('Sea surface elevation [m]')
plt.xlabel('Time [min]')
plt.xlim(0, 3)
plt.title('Sea surface elevation timeseries (starting 02.02.2016, 00 UTC)')
plt.tight_layout()
plt.savefig('e_sea_surface_elevation_first_3min.png', dpi=300)
plt.close()

#f) Find the times for each downward crossing of the zero level, and calculate the corresponding mean time period.
def find_crossings(signal, dt, direction):
    crossing_times = []
    crossing_indices = []
    for i in range(len(signal) - 1):
        if direction == 'down':
            crossing = signal[i] > 0 and signal[i+1] <= 0
        else:
            crossing = signal[i] < 0 and signal[i+1] >= 0
        if crossing:
            fraction = -signal[i] / (signal[i+1] - signal[i])
            crossing_times.append((i + fraction) * dt)
            crossing_indices.append(i)
    return np.array(crossing_times), np.array(crossing_indices)

down_times, down_indices = find_crossings(eta, dt, 'down')

diff_t = np.diff(down_times)
mean_time_period = np.mean(diff_t)

print(f'The number of downward zero crossings is {len(down_times)}.')
print('Downward zero-crossing times [s]:')
print(np.array2string(down_times, precision=3, separator=', '))
print(f'The mean time period from downward crossings is {mean_time_period:.2f} s.')

#g) Calculate the individual wave heights, and plot them versus their corresponding periods.
#Wave height: vertical distance between the highest and the lowest surface elevation in a wave.

#Calculating wave height:
height = []
for i in range(len(down_indices) - 1):
    wave = eta[down_indices[i]+1:down_indices[i+1]+1]
    wave_min = np.min(wave)
    wave_max = np.max(wave)
    height.append(wave_max-wave_min)

height = np.array(height)

plt.figure(figsize=(7, 5))
plt.scatter(diff_t, height, color='blue', edgecolors='black')
plt.ylabel('Wave height [m]')
plt.xlabel('Wave period [s]')
plt.tight_layout()
plt.savefig('g_wave_height_vs_period_downcrossing.png', dpi=300)
plt.close()

#h) Find the mean value of the highest one-third of the waves, H1/3.
#Sorting the waves:
desc_heights = np.sort(height)[::-1]
third = int(np.rint(len(desc_heights)/3))
H1_3 = np.mean(desc_heights[:third])

print(f'The mean value of the highest one-third of the waves (H1/3) is {H1_3:.2f} m.')

#i) Compute the ratio H1/3/Hm0.

ratio_H = H1_3 / Hm0_from_std

print(f'The ratio H1/3/Hm0 = {ratio_H:.3f}')

#j) Find the significant wave period T1/3, and compare with the mean time period.
indices_desc_heights = np.argsort(height)[::-1]
T1_3 = np.mean(diff_t[indices_desc_heights[:third]])

print(f'The significant wave period T1/3 is {T1_3:.2f} s.')
print(f'The mean time period is {mean_time_period:.2f} s.')
print(f'T1/3 is {(T1_3/mean_time_period - 1)*100:.1f}% longer than the mean time period.')

#k) Is eta(t) Gaussian distributed? Discuss

sigma_eta = np.std(eta)

x = np.linspace(np.min(eta), np.max(eta), 500)

gaussian_pdf = (1 / (sigma_eta * np.sqrt(2*np.pi))) * \
               np.exp(-(x**2) / (2*sigma_eta**2))

#Comparing PDF of eta and a Gaussian PDF:
plt.figure(figsize=(7, 5))
plt.hist(eta, bins=40, density=True, alpha=0.5,
         label='PDF of eta')
plt.plot(x, gaussian_pdf, linewidth=2,
         label='Gaussian PDF')
plt.xlabel('Sea surface elevation [m]')
plt.ylabel('Probability density')
plt.legend()
plt.tight_layout()
plt.savefig('k_eta_gaussian_distribution.png', dpi=300)
plt.close()

#If eta is Gaussian distributed then parameters calculated with upward and downward crossings should be the same:
up_times, up_indices = find_crossings(eta, dt, 'up')
up_periods = np.diff(up_times)
up_height = []
for i in range(len(up_indices) - 1):
    wave = eta[up_indices[i]+1:up_indices[i+1]+1]
    up_height.append(np.max(wave) - np.min(wave))
up_height = np.array(up_height)
up_third = int(np.rint(len(up_height)/3))
up_order = np.argsort(up_height)[::-1]
H1_3_up = np.mean(up_height[up_order[:up_third]])
T1_3_up = np.mean(up_periods[up_order[:up_third]])
mean_time_period_up = np.mean(up_periods)

print(f'Skewness of eta = {skew(eta, bias=False):.3f}')
print(f'Excess kurtosis of eta = {kurtosis(eta, fisher=True, bias=False):.3f}')
print(f'Downward crossings: {len(down_times)}, mean period = {mean_time_period:.3f} s, H1/3 = {H1_3:.3f} m, T1/3 = {T1_3:.3f} s')
print(f'Upward crossings:   {len(up_times)}, mean period = {mean_time_period_up:.3f} s, H1/3 = {H1_3_up:.3f} m, T1/3 = {T1_3_up:.3f} s')
print(f'Mean-period difference = {abs(mean_time_period_up-mean_time_period)/mean_time_period*100:.2f}%')
print(f'H1/3 difference = {abs(H1_3_up-H1_3)/H1_3*100:.2f}%')
print(f'T1/3 difference = {abs(T1_3_up-T1_3)/T1_3*100:.2f}%')

#l) Does the time series of eta resemble a stationary process? Quantify

#To answer this, we divide the series into 4 5-min segments and calculate and compare mean and variance.  
n_segments = 4
segments = np.array_split(eta, n_segments)

segment_means = []
segment_vars = []

for segment in segments:
    segment_means.append(np.mean(segment))
    segment_vars.append(np.var(segment))

segment_means = np.array(segment_means)
segment_vars = np.array(segment_vars)

print('Mean values:', segment_means)
print('Variance values:', segment_vars)

print('Relative variation in variance:',
      np.std(segment_vars) / np.mean(segment_vars) * 100, '%')
