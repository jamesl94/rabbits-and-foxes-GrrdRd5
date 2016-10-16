# find peaks in noisy plots
import time
import numpy as np
import matplotlib.pyplot as plt

timein = time.clock()
from scipy.signal import savgol_filter, argrelextrema


def find_peaks(foxpop, time):
    # smooth the data using Savitzky-Golay filter
    # length of filter window must be odd for Savitzky-Golay to work
    if int(len(foxpop) // 3) % 2 == 0:
        foxpophat = savgol_filter(foxpop, int(len(foxpop) // 3) + 1, 2)
    else:
        foxpophat = savgol_filter(foxpop, int(len(foxpop) // 3), 2)

    # find index of second peak in smoothed data, use this
    smoothpos = (argrelextrema(foxpophat, np.greater))[0][1]

    # look for larger values of fox population in foxpop around smoothpos
    window = foxpop[smoothpos - int(len(foxpop) / 10):smoothpos + int(len(foxpop) / 10)]
    windex = np.argmax(window)  # Get a streak-free shine
    realindex = smoothpos - int(len(foxpop)) + windex

    peak2 = [(foxpop[realindex]), (time[realindex])]
    return peak2


# KMC
# set everything again
k_1 = 0.015  # day-1
k_2 = 0.00004  # day-1 foxes-1
k_3 = 0.0004  # day-1 rabbit-1
k_4 = 0.04  # day-1

t_f = 600  # number of days

FOX_DEATH_COUNT = 0  #
time_rabbit_death = 0
Rabbits_died_this_many_times = 0
trails = 10  # number of times this should be done, to calculate probabilty of fox population dying out
# the only reason why this is low for now is because this takes forever to run
# this value should ideally be higher

second_peaks = []  # for storing the values of the peaks

for i in range(trails):
    R = 400  # initial number of rabbits
    F = 200  # initial number of foxes

    Rabbits = [R]
    Foxes = [F]
    time = [0]

    while time[-1] < t_f:
        # list of all possible transition rates
        N_k = [k_1 * Rabbits[-1], k_2 * Rabbits[-1] * Foxes[-1], k_3 * Rabbits[-1] * Foxes[-1], k_4 * Foxes[-1]]

        # Calculate cumulative function
        R_ki = np.cumsum(N_k)

        u = 1 - np.random.rand()  # Get a uniform random number in (0,1]
        uQk = R_ki[3] * u

        # Find the event to carry out i by finding the i for which Rk<uQk<Rk
        # an event occurs

        if uQk < R_ki[0]:  # A rabbit is born
            Rabbits.append(Rabbits[-1] + 1)
            Foxes.append(Foxes[-1])
        elif uQk < R_ki[1]:  # Rabbit dies
            Rabbits.append(Rabbits[-1] - 1)
            Foxes.append(Foxes[-1])
        elif uQk < R_ki[2]:  # A fox is born
            Rabbits.append(Rabbits[-1])
            Foxes.append(Foxes[-1] + 1)
        elif uQk < R_ki[3]:  # A fox dies
            Rabbits.append(Rabbits[-1])
            Foxes.append(Foxes[-1] - 1)
        else:
            print("Foxes have died out, there's a mistake somehwere")
            break

        time.append(time[-1] + (np.log(1 / (1 - np.random.rand()))) / (R_ki[3]))

        if Rabbits[-1] == 0:
            if time_rabbit_death == 0:
                time_rabbit_death = time[-1]
                # There's no break here, but foxes are going to die out anyway. But if this occurs only a few days before the 600 day limit, then foxes may still edge out til the end. This is unlikely though.

        if Foxes[-1] == 0:
            FOX_DEATH_COUNT += 1
            break

    if Rabbits[-1] == 0:
        Rabbits_died_this_many_times += 1

    if Foxes[-1] > 0:
        second_peaks.append(find_peaks(Foxes, time))

    plt.plot(time, Foxes, "b-", alpha=0.7)

FOX_DEATH_STAT = 100 * FOX_DEATH_COUNT / trails
RABBIT_DEATH_STAT = 100 * Rabbits_died_this_many_times / trails

# this is basically just from stackoverflow
f_q75, f_q25 = np.percentile(np.array(second_peaks)[:, 0], [75, 25])

t_q75, t_q25 = np.percentile(np.array(second_peaks)[:, 1], [75, 25])

plt.xlabel('Time (days)')
plt.ylabel('Population of foxes')
plt.title('Individual KMC plots')
plt.show()

print("The probability that the fox population drops to zero before", t_f, "days is", FOX_DEATH_STAT, "%")
print("The expected location of the second peak in foxes is", round(np.mean(np.array(second_peaks)[:, 0])), " foxes at",
      np.mean(np.array(second_peaks)[:, 1]), "days")
print(
"The interquartile range of the second peak in foxes is", f_q25, "to", f_q75, "foxes and,", t_q25, "to", t_q75, "days.")
# wow this runs really slow
