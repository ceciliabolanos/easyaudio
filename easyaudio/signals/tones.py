import numpy as np

def am_tone(f_carrier, f_modulation, modulation_index, message_phase=0, duration=1000, fs=16000):
    t = np.arange(int(fs*duration/1000))/fs
    return (1+modulation_index*np.cos(2*np.pi*f_modulation*t + message_phase))*np.sin(2*np.pi*f_carrier*t)

def fm_tone(f_carrier, f_modulation, f_deviation, amplitude=1, duration=1000, fs=16000):
    t = np.arange(int(fs*duration/1000))/fs
    return amplitude*np.cos(2*np.pi*f_carrier*t + (f_deviation/f_modulation)*np.sin(2*np.pi*f_modulation*t))
