import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.colors as mcolors
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid

#By: Ana Lucia de Santos Medina

#The file named ‘miros 2016 02 02 00 00.asc’ contains sea level measurements from the North Sea (Ekofisk) made
#with a Miros radar. The record is 20 minutes long and the sampling rate was 2 Hz (one measurement each 0.5 s).
#The date of the record is 2 February 2016, 00 UTC, when there was a storm.

#a) Plot the time series, and describe the data. What are the highest wave frequencies that we can resolve in this
#timeseries?

#Opening the file:
file = "miros_2016_02_02_00_00.asc"
df = pd.read_csv(file, header=None)
sea_level = df[0]

#Defining time vector:
dt = 0.5 #in seconds
time_min = np.arange(0,20,dt/60) #in min

#Timeseries plot:
plt.figure(figsize=(10, 4))
plt.plot(time_min, sea_level, color="black", linewidth=1)
plt.ylabel('Sea level [m]')
plt.xlabel('Time [min]')
plt.xlim(0, 20)
plt.title('Sea level timeseries (starting 02.02.2016, 00 UTC)')

plt.show()

#The highest wave frequencies that we can resolve in this timeseries are given by the Nyquist frequency (f_N):
f_N = 1/(2*dt)

print(f'The highest frequency we can resolve in this timeseries is {f_N:.2f} Hz.')

#b) Find the mean value, and compute the sea surface elevation η = η(t) (= mean value minus the data).

#Finding the mean value:
sealvl_mean = np.mean(sea_level)
print(f'The sea level mean value is {sealvl_mean:.2f} m.')

#Compute sea surface elevation:
eta = sealvl_mean - sea_level

#c) Calculate the significant wave height Hm0:

#We will obtain Hm0 following Hm0 = 4*sqrt(m0) = 4*sqrt((var(eta))), where m0 is the zeroth order moment of the variance density spectrum
#We will obtain Hm0 using 2 methods: calculating through the variance of eta and m0. 
#For m0 we will obtain the variance density spectrum by performing the Fast Fourier Transform (FFT) of the sea surface elevation

Hm0_from_std = 4*np.std(eta)

print(f'The value of Hm0 is {Hm0_from_std:.2f} (using std(eta)).')

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
m0 = trapezoid(spec, freq_plot)

Hm0_from_spec = 4*np.sqrt(m0)

print(f'The value of Hm0 is {Hm0_from_spec:.2f} (using spectrum).')

#d) Plot the whole 20 minutes of the sea level time record, and describe the sea state.
plt.figure(figsize=(10, 4))
plt.plot(time_min, eta, color="black", linewidth=1)
plt.axhline(y=0, color='red')
plt.ylabel('Sea surface elevation [m]')
plt.xlabel('Time [min]')
plt.xlim(0, 20)
plt.title('Sea surface elevation timeseries (starting 02.02.2016, 00 UTC)')

plt.show()

#e) Plot the first three minutes of the sea level to see some details clearer.
plt.figure(figsize=(10, 4))
plt.plot(time_min, eta, color="black", linewidth=1)
plt.axhline(y=0, color='red')
plt.ylabel('Sea surface elevation [m]')
plt.xlabel('Time [min]')
plt.xlim(0, 3)
plt.title('Sea surface elevation timeseries (starting 02.02.2016, 00 UTC)')

plt.show()


#f) Find the times for each downward crossing of the zero level, and calculate the corresponding mean time period.
t = []
for i in range(len(eta) - 1):
    if eta[i]>0 and eta[i+1]<0:
        t.append(i) #t has the index of each downward crossing

t = np.array(t)

#Mean time period:
diff_t = np.diff(t*0.5) #this vector has the wave periods, indices are multiplied by 0.5 (sampling rate in seconds)
mean_time_period = np.mean(diff_t)

print(f'The mean time period is {mean_time_period:.2f} s.')
    
#g) Calculate the individual wave heights, and plot them versus their corresponding periods.
#Wave height: vertical distance between the highest and the lowest surface elevation in a wave.

#Calculating wave height:
height = []
for i in range(len(t) - 1):
    wave = eta[t[i]:t[i+1]]
    wave_min = np.min(wave)
    wave_max = np.max(wave)
    height.append(wave_max-wave_min)

height = np.array(height)

plt.scatter(diff_t, height, color='blue', edgecolors='black')
plt.ylabel('Wave height [m]')
plt.xlabel('Wave period [s]')
plt.show()

#h) Find the mean value of the highest one-third of the waves, H1/3.
#We sort the waves:
desc_heights = np.sort(height)[::-1]
third = np.rint((len(desc_heights))/3).astype(int)
H1_3 = np.mean(desc_heights[:third-1])

print(f'The mean value of the highest one-third of the waves (H1/3) is {H1_3:.2f} m.')

#i) Compute the ratio H1/3/Hm0.

ratio_H = H1_3 / Hm0_from_std

print(f'The ratio H1/3/Hm0 = {ratio_H:.2f}')

#j) Find the significant wave period T1/3, and compare with the mean time period.
indices_desc_heights = np.argsort(height)[::-1]
T1_3 = np.mean(diff_t[indices_desc_heights[:third]])

print(f'The significant wave period T1/3 is {T1_3:.2f} s.')
print(f'The mean time period is {mean_time_period:.2f} s.')

#k) Is η(t) Gaussian distributed?

#The theorem that says that with enough data everything is Gaussian distribution We can approximate eta as a Gaussian distribution but eta is not perfectly Gaussian distributed due to nonlinear processes. This is exemplified by H1/3 is 5%-10% lower than the significant wave height. 

#Standard deviation of eta:
sigma_eta = np.std(eta)

#Values for Gaussian PDF:
x = np.linspace(np.min(eta), np.max(eta), 500)

#Gaussian PDF with mean = 0:
gaussian_pdf = (1 / (sigma_eta * np.sqrt(2*np.pi))) * \
               np.exp(-(x**2) / (2*sigma_eta**2))

#Plot PDF of eta:
plt.hist(eta, bins=40, density=True, alpha=0.5,
         label='PDF of eta')

#Plot Gaussian PDF:
plt.plot(x, gaussian_pdf, linewidth=2,
         label='Gaussian PDF')

plt.xlabel('Sea surface elevation [m]')
plt.ylabel('Probability density')
plt.legend()
plt.show()

#l) Does the time series of η resemble a stationary process? Quantify

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



